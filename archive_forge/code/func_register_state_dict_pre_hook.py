from collections import OrderedDict, namedtuple
import itertools
import warnings
import functools
import weakref
import torch
from torch._prims_common import DeviceLikeType
from ..parameter import Parameter
import torch.utils.hooks as hooks
from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from typing_extensions import Self
from ...utils.hooks import RemovableHandle
def register_state_dict_pre_hook(self, hook):
    """Register a pre-hook for the :meth:`~torch.nn.Module.load_state_dict` method.

        These hooks will be called with arguments: ``self``, ``prefix``,
        and ``keep_vars`` before calling ``state_dict`` on ``self``. The registered
        hooks can be used to perform pre-processing before the ``state_dict``
        call is made.
        """
    handle = hooks.RemovableHandle(self._state_dict_pre_hooks)
    self._state_dict_pre_hooks[handle.id] = hook
    return handle