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
@register_op_impl(aten.index_put.default)
@register_op_impl(aten.index_put_.default)
def index_put_impl(fake_mode, func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    values = new_kwargs['values']
    self_device = new_kwargs['input'].fake_device
    torch._check(self_device == values.fake_device or (values.ndim == 0 and values.numel() == 1), lambda: f'Mismatching {func} device between self ({self_device}) and values ({values.device})')
    out = run_and_return_new_tensor_of_input_device(fake_mode, func, args, kwargs)
    if func is aten.index_put_.default:
        return new_kwargs['input']
    else:
        return out