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
@register_op_impl(lambda func: func is aten.repeat_interleave.Tensor)
def repeat_interleave_tensor(fake_mode, func, repeats, output_size=None):
    if output_size is None:
        if fake_mode.shape_env is None or not fake_mode.shape_env.allow_dynamic_output_shape_ops:
            raise DynamicOutputShapeException(func)
        output_size = fake_mode.shape_env.create_unbacked_symint()
        from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size
        _constrain_range_for_size(output_size)
    return repeats.new_empty(output_size)