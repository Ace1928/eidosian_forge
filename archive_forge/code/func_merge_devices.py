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
def merge_devices(t):
    nonlocal common_device
    nonlocal is_cpu_zero_dim
    if not isinstance(t, FakeTensor):
        return
    if common_device is None:
        common_device = t.device
        is_cpu_zero_dim = cpu_zero_dim(t)
        return
    t_is_cpu_zero_dim = cpu_zero_dim(t)
    if t.device == common_device:
        if is_cpu_zero_dim:
            is_cpu_zero_dim = t_is_cpu_zero_dim
        return
    if t_is_cpu_zero_dim:
        return
    if is_cpu_zero_dim:
        common_device = t.device
        is_cpu_zero_dim = t_is_cpu_zero_dim
        return
    raise RuntimeError(f'Unhandled FakeTensor Device Propagation for {func}, found two different devices {common_device}, {t.device}')