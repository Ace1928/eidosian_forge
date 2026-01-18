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
def quantize_net(network, quantized_dtype='auto', quantize_mode='full', exclude_layers=None, exclude_layers_match=None, exclude_operators=None, calib_data=None, data_shapes=None, calib_mode='none', num_calib_examples=None, ctx=cpu(), logger=None):
    """User-level API for Gluon users to generate a quantized SymbolBlock from a FP32 HybridBlock w/ or w/o calibration.
       Will be deprecated after MXNet 2.0, please use quantize_net_v2.
    """
    warnings.warn('WARNING: This will be deprecated after MXNet 2.0, please use quantize_net_v2.')
    return quantize_net_v2(network=network, quantized_dtype=quantized_dtype, quantize_mode=quantize_mode, quantize_granularity='tensor-wise', exclude_layers=exclude_layers, exclude_layers_match=exclude_layers_match, exclude_operators=exclude_operators, calib_data=calib_data, data_shapes=data_shapes, calib_mode=calib_mode, num_calib_examples=num_calib_examples, ctx=ctx, LayerOutputCollector=None, logger=logger)