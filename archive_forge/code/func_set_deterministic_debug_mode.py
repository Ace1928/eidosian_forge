import math
import os
import sys
import platform
import textwrap
import ctypes
import inspect
from ._utils import _import_dotted_name, classproperty
from ._utils import _functionalize_sync as _sync
from ._utils_internal import get_file_path, prepare_multiprocessing_environment, \
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, TYPE_CHECKING, Union, List
import builtins
from math import e , nan , inf , pi
from ._tensor import Tensor
from .storage import _StorageBase, TypedStorage, _LegacyStorage, UntypedStorage, _warn_typed_storage_removal
from .random import set_rng_state, get_rng_state, manual_seed, initial_seed, seed
from .serialization import save, load
from ._tensor_str import set_printoptions
from torch.amp import autocast
from ._compile import _disable_dynamo
from .functional import *  # noqa: F403
from torch import cuda as cuda
from torch import cpu as cpu
from torch import mps as mps
from torch import autograd as autograd
from torch.autograd import (
from torch import fft as fft
from torch import futures as futures
from torch import _awaits as _awaits
from torch import nested as nested
from torch import nn as nn
from torch.signal import windows as windows
from torch import optim as optim
import torch.optim._multi_tensor
from torch import multiprocessing as multiprocessing
from torch import sparse as sparse
from torch import special as special
import torch.utils.backcompat
from torch import jit as jit
from torch import linalg as linalg
from torch import hub as hub
from torch import random as random
from torch import distributions as distributions
from torch import testing as testing
from torch import backends as backends
import torch.utils.data
from torch import __config__ as __config__
from torch import __future__ as __future__
from torch import profiler as profiler
from torch import ao as ao
import torch.nn.quantizable
import torch.nn.quantized
import torch.nn.qat
import torch.nn.intrinsic
from . import _torch_docs, _tensor_docs, _storage_docs
from torch._ops import ops
from torch._classes import classes
import torch._library
from torch import quantization as quantization
from torch import quasirandom as quasirandom
from torch.multiprocessing._atfork import register_after_fork
from ._lobpcg import lobpcg as lobpcg
from torch.utils.dlpack import from_dlpack, to_dlpack
from . import masked
from ._linalg_utils import (  # type: ignore[misc]
from ._linalg_utils import _symeig as symeig  # type: ignore[misc]
from torch import export as export
from torch._higher_order_ops import cond
from . import return_types
from . import library
import torch.fx.experimental.sym_node
from torch import func as func
from torch.func import vmap
from . import _logging
def set_deterministic_debug_mode(debug_mode: Union[builtins.int, str]) -> None:
    """Sets the debug mode for deterministic operations.

    .. note:: This is an alternative interface for
        :func:`torch.use_deterministic_algorithms`. Refer to that function's
        documentation for details about affected operations.

    Args:
        debug_mode(str or int): If "default" or 0, don't error or warn on
            nondeterministic operations. If "warn" or 1, warn on
            nondeterministic operations. If "error" or 2, error on
            nondeterministic operations.
    """
    if not isinstance(debug_mode, (builtins.int, str)):
        raise TypeError(f'debug_mode must be str or int, but got {type(debug_mode)}')
    if isinstance(debug_mode, str):
        if debug_mode == 'default':
            debug_mode = 0
        elif debug_mode == 'warn':
            debug_mode = 1
        elif debug_mode == 'error':
            debug_mode = 2
        else:
            raise RuntimeError(f'invalid value of debug_mode, expected one of `default`, `warn`, `error`, but got {debug_mode}')
    if debug_mode == 0:
        _C._set_deterministic_algorithms(False)
    elif debug_mode == 1:
        _C._set_deterministic_algorithms(True, warn_only=True)
    elif debug_mode == 2:
        _C._set_deterministic_algorithms(True)
    else:
        raise RuntimeError(f'invalid value of debug_mode, expected 0, 1, or 2, but got {debug_mode}')