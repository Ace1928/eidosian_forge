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
def quantize_net_v2(network, quantized_dtype='auto', quantize_mode='full', quantize_granularity='tensor-wise', exclude_layers=None, exclude_layers_match=None, exclude_operators=None, calib_data=None, data_shapes=None, calib_mode='none', num_calib_examples=None, ctx=cpu(), LayerOutputCollector=None, logger=None):
    """User-level API for Gluon users to generate a quantized SymbolBlock from a FP32 HybridBlock w/ or w/o calibration.
    The backend quantized operators are only enabled for Linux systems. Please do not run
    inference using the quantized models on Windows for now.

    Parameters
    ----------
    network : Gluon HybridBlock
        Defines the structure of a neural network for FP32 data types.
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
    exclude_layers : list of strings
        A list of strings representing the names of the symbols that users want to excluding
    exclude_layers_match : list of strings
        A list of strings wildcard matching the names of the symbols that users want to excluding
        from being quantized.
    exclude_operators : list of strings
        A list of strings representing the names of the operators that users want to excluding
    calib_data : mx.io.DataIter or gluon.DataLoader
        A iterable data loading object.
    data_shapes : list
        List of DataDesc, required if calib_data is not provided
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
    num_calib_examples : int or None
        The maximum number of examples that user would like to use for calibration. If not provided,
        the whole calibration dataset will be used.
    ctx : Context
        Defines the device that users want to run forward propagation on the calibration
        dataset for collecting layer output statistics. Currently, only supports single context.
    LayerOutputCollector : class
        For customize calibration method usage.
    logger : Object
        A logging object for printing information during the process of quantization.

    Returns
    -------
    network : Gluon SymbolBlock
        Defines the structure of a neural network for INT8 data types.
    -------
    """
    if logger:
        logger.info('Export HybridBlock')
    network.hybridize()
    import mxnet as mx
    if calib_data is not None:
        if isinstance(calib_data, DataIter):
            dshapes = calib_data.provide_data
        else:
            calib_data, dshapes = _as_data_iter(calib_data)
        if not data_shapes:
            data_shapes = dshapes
    if not data_shapes:
        raise ValueError('data_shapes required')
    data_nd = []
    for shape in data_shapes:
        data_nd.append(mx.nd.zeros(shape.shape))
    while True:
        try:
            network(*data_nd)
        except TypeError:
            del data_nd[-1]
            del calib_data.provide_data[-1]
            continue
        else:
            break
    import tempfile
    try:
        from tempfile import TemporaryDirectory
    except ImportError:

        class TemporaryDirectory(object):

            def __init__(self, suffix='', prefix='', dir=''):
                self._dirname = tempfile.mkdtemp(suffix, prefix, dir)

            def __enter__(self):
                return self._dirname

            def __exit__(self, exc_type, exc_value, traceback):
                shutil.rmtree(self._dirname)
    with TemporaryDirectory() as tmpdirname:
        prefix = os.path.join(tmpdirname, 'tmp')
        network.export(prefix, epoch=0)
        symnet, args, auxs = mx.model.load_checkpoint(prefix, 0)
    if exclude_layers is None:
        exclude_layers = []
    if exclude_layers_match is None:
        exclude_layers_match = []
    if exclude_operators is None:
        exclude_operators = []
    for name_match in exclude_layers_match:
        for layers in list(symnet.get_internals()):
            if layers.name.find(name_match) != -1:
                exclude_layers.append(layers.name)
    if logger:
        logger.info('These layers have been excluded %s' % exclude_layers)
    if ctx == mx.cpu():
        symnet = symnet.get_backend_symbol('MKLDNN_QUANTIZE')
    qsym, qarg_params, aux_params, collector = quantize_graph(sym=symnet, arg_params=args, aux_params=auxs, ctx=ctx, excluded_sym_names=exclude_layers, excluded_op_names=exclude_operators, calib_mode=calib_mode, quantized_dtype=quantized_dtype, quantize_mode=quantize_mode, quantize_granularity=quantize_granularity, LayerOutputCollector=LayerOutputCollector, logger=logger)
    if calib_mode is not None and calib_mode != 'none':
        if not isinstance(ctx, Context):
            raise ValueError('currently only supports single ctx, while received %s' % str(ctx))
        if calib_data is None:
            raise ValueError('calib_data must be provided when calib_mode=%s' % calib_mode)
        if calib_mode in ['naive', 'entropy', 'customize']:
            data_names = [pair[0] for pair in calib_data.provide_data]
            mod = Module(symbol=symnet, context=ctx, data_names=data_names, label_names=None)
            mod.bind(for_training=False, data_shapes=data_shapes)
            mod.set_params(args, auxs, allow_missing=False, force_init=True)
            num_examples = _collect_layer_statistics(mod, calib_data, collector, num_calib_examples, logger)
            if logger:
                logger.info('Collected layer output values from FP32 model using %d examples' % num_examples)
            qsym, qarg_params, aux_params = calib_graph(qsym=qsym, arg_params=args, aux_params=auxs, collector=collector, calib_mode=calib_mode, quantized_dtype=quantized_dtype, logger=logger)
        else:
            raise ValueError('please set calibration mode to naive or entropy.')
    elif calib_mode is not None and calib_mode == 'none':
        data_names = [pair[0] for pair in data_shapes]
    if ctx == mx.cpu():
        qsym = qsym.get_backend_symbol('MKLDNN_QUANTIZE')
    from ..gluon import SymbolBlock
    data_sym = []
    for name in data_names:
        data_sym.append(mx.sym.var(name))
    net = SymbolBlock(qsym, data_sym)
    with TemporaryDirectory() as tmpdirname:
        prefix = os.path.join(tmpdirname, 'tmp')
        param_name = '%s-%04d.params' % (prefix + 'net-quantized', 0)
        save_dict = {'arg:%s' % k: v.as_in_context(cpu()) for k, v in qarg_params.items()}
        save_dict.update({'aux:%s' % k: v.as_in_context(cpu()) for k, v in aux_params.items()})
        nd_save(param_name, save_dict)
        net.collect_params().load(param_name, cast_dtype=True, dtype_source='saved')
        net.collect_params().reset_ctx(ctx)
    return net