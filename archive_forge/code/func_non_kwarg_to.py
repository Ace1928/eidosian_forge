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
@register_op_impl(lambda func: func in (aten.to.prim_Device, aten.to.device))
def non_kwarg_to(fake_mode, func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args, kwargs, normalize_to_only_use_kwargs=True)
    input_device = new_kwargs['device']
    out_device = input_device if input_device else new_kwargs['input'].device
    new_kwargs['device'] = torch.device('meta')
    inp = new_kwargs.pop('input')
    with in_kernel_invocation_manager(fake_mode):
        r = func(inp, **new_kwargs)
    return fake_mode.fake_tensor_converter.from_meta_and_device(fake_mode, r, out_device)