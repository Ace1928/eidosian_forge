import torch
import inspect
import numbers
import types
import typing
import enum
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, cast, TYPE_CHECKING
from torch._jit_internal import boolean_dispatched
from ._compatibility import compatibility
from torch._ops import OpOverloadPacket, OpOverload
@compatibility(is_backward_compatible=False)
def create_type_hint(x):
    try:
        if isinstance(x, (list, tuple)):
            if isinstance(x, list):

                def ret_type(x):
                    return List[x]
            else:

                def ret_type(x):
                    return Tuple[x, ...]
            if len(x) == 0:
                return ret_type(Any)
            base_type = x[0]
            for t in x:
                if issubclass(t, base_type):
                    continue
                elif issubclass(base_type, t):
                    base_type = t
                else:
                    return ret_type(Any)
            return ret_type(base_type)
    except Exception as e:
        warnings.warn(f'We were not able to successfully create type hint from the type {x}')
        pass
    return x