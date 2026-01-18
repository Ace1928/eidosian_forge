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
def invalidate_written_to_constants(self, func, flat_arg_fake_tensors, args, kwargs):
    any_constant = any((e.constant is not None for e in flat_arg_fake_tensors))
    schema_info = get_schema_info(func)
    if any_constant and schema_info.is_mutable():
        _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
        for k, v in new_kwargs.items():
            k = k if k != 'input' or schema_info.has_argument(k) else 'self'
            if self.is_our_fake(v) and schema_info.is_mutable(k) and (v.constant is not None):
                self.fake_tensor_converter.invalidate_constant_aliases(v.constant)