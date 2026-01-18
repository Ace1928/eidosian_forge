from array import array
import ctypes
import logging
import contextlib
import numpy as np
from ... import symbol
from ...context import gpu
from ...symbol import Symbol
from ...module import BucketingModule
from ...symbol import contrib as symbol_contrib
from ... import ndarray
from ...ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from . import lists
from ...gluon import trainer
from ... import base
from ...base import c_str_array, SymbolHandle, check_call, _LIB, mx_uint, c_array_buf
from ... import optimizer as opt
from .loss_scaler import LossScaler
def list_fp32_ops(target_dtype):
    """Get the default list of FP32 ops for AMP
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.FP32_FUNCS
    else:
        assert target_dtype == bfloat16, 'not supported type'
        return lists.symbol_bf16.FP32_FUNCS