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
def invalidate_constant_aliases(self, tensor):
    assert not isinstance(tensor, FakeTensor)
    weak_st = StorageWeakRef(tensor._typed_storage())
    if weak_st not in self.constant_storage_mapping:
        return
    for weak_tensor_ref in self.constant_storage_mapping[weak_st]:
        ten = weak_tensor_ref()
        if ten is not None:
            ten._fix_weakref()
            ten.constant = None
    del self.constant_storage_mapping[weak_st]