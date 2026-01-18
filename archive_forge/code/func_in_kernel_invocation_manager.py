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
@contextlib.contextmanager
def in_kernel_invocation_manager(fake_mode):
    prev_in_kernel = fake_mode.in_kernel_invocation
    meta_in_tls = torch._C._meta_in_tls_dispatch_include()
    assert meta_in_tls == prev_in_kernel, f'{meta_in_tls}, {prev_in_kernel}'
    guard = torch._C._DisableTorchDispatch()
    fake_mode.in_kernel_invocation = True
    torch._C._set_meta_in_tls_dispatch_include(True)
    try:
        yield
    finally:
        fake_mode.in_kernel_invocation = prev_in_kernel
        torch._C._set_meta_in_tls_dispatch_include(prev_in_kernel)
        del guard