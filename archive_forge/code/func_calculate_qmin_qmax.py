import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized
def calculate_qmin_qmax(quant_min: int, quant_max: int, has_customized_qrange: bool, dtype: torch.dtype, reduce_range: bool) -> Tuple[int, int]:
    """Calculates actual qmin and qmax based on the quantization range,
    observer datatype and if range is reduced.
    """
    if has_customized_qrange:
        if dtype in [torch.qint32, torch.int32]:
            initial_quant_min, initial_quant_max = (0, 2 ** 32 - 1)
        else:
            initial_quant_min, initial_quant_max = (0, 255)
        custom_quant_min, custom_quant_max = (quant_min, quant_max)
        if custom_quant_min is not None and custom_quant_max is not None:
            initial_quant_min, initial_quant_max = (custom_quant_min, custom_quant_max)
        qrange_len = initial_quant_max - initial_quant_min + 1
        if dtype in [torch.qint8, torch.int8]:
            assert 0 < qrange_len <= 256, 'quantization range should be positive and not exceed the maximum bit range (=256).'
        elif dtype in [torch.qint32, torch.int32]:
            assert 0 < qrange_len <= 2 ** 32, 'quantization range should be positive and not exceed the maximum bit range (=4294967296).'
        if reduce_range:
            quant_min, quant_max = (quant_min // 2, quant_max // 2)
    elif dtype in [torch.qint8, torch.int8]:
        if reduce_range:
            quant_min, quant_max = (-64, 63)
        else:
            quant_min, quant_max = (-128, 127)
    elif dtype in [torch.quint8, torch.uint8]:
        if reduce_range:
            quant_min, quant_max = (0, 127)
        else:
            quant_min, quant_max = (0, 255)
    elif dtype in [torch.qint32, torch.int32]:
        quant_min, quant_max = (-1 * 2 ** 31, 2 ** 31 - 1)
    else:
        quant_min, quant_max = (0, 15)
    return (quant_min, quant_max)