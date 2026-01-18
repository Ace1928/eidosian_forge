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
def remove_from(*dicts_or_sets):
    for d in dicts_or_sets:
        if name in d:
            if isinstance(d, dict):
                del d[name]
            else:
                d.discard(name)