import contextlib
import functools
import itertools
import logging
import os
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from weakref import ReferenceType
import torch
import torch._custom_op
import torch._logging
from torch._guards import Source
from torch._ops import OpOverload
from torch._prims_common import (
from torch._subclasses.meta_utils import MetaConverter
from torch._utils import render_call
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
from torch.utils._pytree import PyTree, tree_map
from torch.utils._stats import count, count_label
from torch.utils.weak import WeakIdRef
@register_op_impl(lambda func: _is_tensor_constructor(func) or func in _like_tensor_constructors)
def constructors(fake_mode, func, *args, **kwargs):
    assert func not in _non_kwarg_device_constructors
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    if func in _like_tensor_constructors:
        default_device = new_kwargs['input'].device
        args = (new_kwargs.pop('input'),)
    else:
        default_device = torch.device('cpu')
        args = ()
    out_device = new_kwargs.pop('device', None)
    out_device = out_device if out_device is not None else default_device
    new_kwargs['device'] = torch.device('meta')
    with in_kernel_invocation_manager(fake_mode):
        r = func(*args, **new_kwargs)
    return FakeTensor(fake_mode, r, out_device)