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
def is_fake(x):
    if isinstance(x, FakeTensor):
        return True
    if is_traceable_wrapper_subclass(x):
        attrs, _ = type(x).__tensor_flatten__(x)
        flattened_tensors = [getattr(x, attr) for attr in attrs]
        all_fake = all((is_fake(x) for x in flattened_tensors))
        any_fake = any((is_fake(x) for x in flattened_tensors))
        assert all_fake == any_fake, 'got mixed fake and real tensors!'
        return all_fake
    elif isinstance(x, torch.Tensor) and torch._is_functional_tensor(x):
        reapply_views = torch._C._functionalization_reapply_views_tls()
        unwrapped = torch._C._functorch._unwrap_functional_tensor(x, reapply_views)
        return is_fake(unwrapped)
    return False