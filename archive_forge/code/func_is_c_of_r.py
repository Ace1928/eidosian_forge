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
def is_c_of_r(complex_dtype, real_dtype):
    return utils.is_complex_dtype(complex_dtype) and utils.corresponding_real_dtype(complex_dtype) == real_dtype