import ctypes
import logging
import os
import shutil
import warnings
import numpy as np
from ..base import _LIB, check_call, py_str
from ..base import c_array, c_str, mx_uint, c_str_array
from ..base import NDArrayHandle, SymbolHandle
from ..symbol import Symbol
from ..symbol import load as sym_load
from .. import ndarray
from ..ndarray import load as nd_load
from ..ndarray import save as nd_save
from ..ndarray import NDArray
from ..io import DataIter, DataDesc, DataBatch
from ..context import cpu, Context
from ..module import Module
def quantize_graph(sym, arg_params, aux_params, ctx=cpu(), excluded_sym_names=None, excluded_op_names=None, calib_mode='entropy', quantized_dtype='int8', quantize_mode='full', quantize_granularity='tensor-wise', LayerOutputCollector=None, logger=None):
    """User-level API for generating a quantized model from a FP32 model w/o calibration
    and a collector for naive or entropy calibration.
    The backend quantized operators are only enabled for Linux systems. Please do not run
    inference using the quantized models on Windows for now.
    Parameters
    ----------
    sym : str or Symbol
        Defines the structure of a neural network for FP32 data types.
    ctx : Context
        Defines the device that users want to run forward propagation on the calibration
        dataset for collecting layer output statistics. Currently, only supports single context.
    arg_params : dict
        Dictionary of name to `NDArray`.
    aux_params : dict
        Dictionary of name to `NDArray`.
    excluded_sym_names : list of strings
        A list of strings representing the names of the symbols that users want to excluding
        from being quantized.
    excluded_op_names : list of strings
        A list of strings representing the names of the operators that users want to excluding
    calib_mode : str
        If calib_mode='none', no calibration will be used and the thresholds for
        requantization after the corresponding layers will be calculated at runtime by
        calling min and max operators. The quantized models generated in this
        mode are normally 10-20% slower than those with calibrations during inference.
        If calib_mode='naive', the min and max values of the layer outputs from a calibration
        dataset will be directly taken as the thresholds for quantization.
        If calib_mode='entropy' (default mode), the thresholds for quantization will be
        derived such that the KL divergence between the distributions of FP32 layer outputs and
        quantized layer outputs is minimized based upon the calibration dataset.
    quantized_dtype : str
        The quantized destination type for input data. Currently support 'int8'
        , 'uint8' and 'auto'. 'auto' means automatically select output type according to calibration result.
        Default value is 'int8'.
    quantize_mode : str
        The mode that quantization pass to apply. Support 'full' and 'smart'.
        'full' means quantize all operator if possible.
        'smart' means quantization pass will smartly choice which operator should be quantized.
    quantize_granularity: str
        The granularity of quantization, currently supports 'tensor-wise' and 'channel-wise'
        quantization. The default value is 'tensor-wise'.
    LayerOutputCollector : class
        For customize calibration method usage.
    logger : Object
        A logging object for printing information during the process of quantization.
    Returns
    -------
    tuple
        A tuple of quantized symbol, quantized arg_params, aux_params and collector.
    -------
    """
    if excluded_sym_names is None:
        excluded_sym_names = []
    if not isinstance(excluded_sym_names, list):
        raise ValueError('excluded_sym_names must be a list of strings representing the names of the symbols that will not be quantized, while received type %s' % str(type(excluded_sym_names)))
    if not isinstance(ctx, Context):
        raise ValueError('currently only supports single ctx, while received %s' % str(ctx))
    if logger:
        os.environ['MXNET_QUANTIZATION_VERBOSE'] = '1'
        logger.info('Quantizing graph')
    if quantized_dtype not in ('int8', 'uint8', 'auto'):
        raise ValueError('unknown quantized_dtype %s received, expected `int8`, `uint8` or `auto`' % quantized_dtype)
    if quantize_granularity not in ('tensor-wise', 'channel-wise'):
        raise ValueError('unkonwn quantize_granularity %s received, expected `tensor-wise` or `channel-wise`.' % quantize_granularity)
    qsym, calib_layer = _quantize_symbol(sym, ctx, excluded_symbols=excluded_sym_names, excluded_operators=excluded_op_names, offline_params=list(arg_params.keys()), quantized_dtype=quantized_dtype, quantize_mode=quantize_mode, quantize_granularity=quantize_granularity)
    th_dict = {}
    collector = None
    if calib_mode is not None and calib_mode != 'none':
        if calib_mode == 'entropy':
            collector = _LayerHistogramCollector(include_layer=calib_layer, logger=logger)
            if logger:
                logger.info('Create a layer output collector for entropy calibration.')
        elif calib_mode == 'naive':
            collector = _LayerOutputMinMaxCollector(quantized_dtype=quantized_dtype, include_layer=calib_layer, logger=logger)
            if logger:
                logger.info('Create a layer output minmax collector for naive calibration')
        elif calib_mode == 'customize' and LayerOutputCollector is not None:
            collector = LayerOutputCollector
            if logger:
                logger.info('Create a customize layer output minmax collector for calibration')
        else:
            raise ValueError('unknown calibration mode %s received, expected `none`, `naive`, `entropy` or `customize`' % calib_mode)
        if logger:
            logger.info('Collector created, please use set_monitor_callback to collect calibration information.')
    if logger:
        logger.info('Quantizing parameters')
    qarg_params = _quantize_params(qsym, arg_params, th_dict)
    return (qsym, qarg_params, aux_params, collector)