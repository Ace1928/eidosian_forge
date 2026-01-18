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
def cpp_meta_supports_symint(self, func):
    if torch.Tag.view_copy in func.tags:
        return True
    return func in [aten.empty.memory_format, aten.empty_strided.default, aten.as_strided_scatter.default, aten.as_strided.default, aten.as_strided_.default, aten.zeros.default, aten.detach.default, aten.view_as_real.default, aten.view_as_complex.default, aten.set_.source_Storage_storage_offset, aten._sparse_coo_tensor_with_dims_and_tensors.default]