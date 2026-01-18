import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized
def get_quant_type(qconfig):
    assert qconfig is not None
    activation = qconfig.activation()
    weight = qconfig.weight()
    static_dtypes = [torch.quint8, torch.qint8, torch.quint4x2, torch.qint32, torch.uint8, torch.int8, torch.int16, torch.int32]
    if weight.dtype in static_dtypes:
        if hasattr(activation, 'is_dynamic') and activation.is_dynamic:
            return QuantType.DYNAMIC
        elif activation.dtype in static_dtypes:
            return QuantType.STATIC
        else:
            return QuantType.WEIGHT_ONLY
    if weight.dtype == torch.float16:
        if hasattr(activation, 'is_dynamic') and activation.is_dynamic:
            return QuantType.DYNAMIC
        elif activation.dtype == torch.float16:
            return QuantType.STATIC
    raise Exception(f'Unrecognized dtype combination in get_quant_type: activation({activation.dtype}),weight({weight.dtype})')