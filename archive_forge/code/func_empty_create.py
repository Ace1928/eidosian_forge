import contextlib
import warnings
import weakref
from typing import ContextManager, List, Optional, Tuple, TYPE_CHECKING
import torch
from torch._C._functorch import (
from torch._guards import Source
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from torch.utils.weak import WeakIdRef
import torch._prims_common as utils
def empty_create(inner_t, inner_src):
    inner_sizes, inner_strides, inner_storage_offset = sym_sizes_strides_storage_offset(inner_t, inner_src)
    return torch.empty_strided(inner_sizes, inner_strides, dtype=inner_t.dtype, device='meta')