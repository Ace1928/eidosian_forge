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
def quantize_model_mkldnn(sym, arg_params, aux_params, data_names=('data',), label_names=('softmax_label',), ctx=cpu(), excluded_sym_names=None, excluded_op_names=None, calib_mode='entropy', calib_data=None, num_calib_examples=None, quantized_dtype='int8', quantize_mode='smart', quantize_granularity='tensor-wise', logger=None):
    """User-level API for generating a fusion + quantized model from a FP32 model
    w/ or w/o calibration with Intel MKL-DNN.
    The backend quantized operators are only enabled for Linux systems. Please do not run
    inference using the quantized models on Windows for now.

    Parameters
    ----------
    same with quantize_model

    Returns
    -------
    tuple
        A tuple of quantized symbol, quantized arg_params, and aux_params.
    -------
    """
    if not isinstance(ctx, Context):
        raise ValueError('currently only supports single ctx, while received %s' % str(ctx))
    if ctx.device_type != 'cpu':
        raise ValueError('quantize_model_mkldnn only support Intel cpu platform with MKL-DNN Backend')
    sym = sym.get_backend_symbol('MKLDNN_QUANTIZE')
    qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params, data_names=data_names, label_names=label_names, ctx=ctx, excluded_sym_names=excluded_sym_names, excluded_op_names=excluded_op_names, calib_mode=calib_mode, calib_data=calib_data, num_calib_examples=num_calib_examples, quantized_dtype=quantized_dtype, quantize_mode=quantize_mode, quantize_granularity=quantize_granularity, logger=logger)
    qsym = qsym.get_backend_symbol('MKLDNN_QUANTIZE')
    return (qsym, qarg_params, aux_params)