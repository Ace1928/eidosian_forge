import copyreg
import enum
import functools
import warnings
from collections import OrderedDict
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._namedtensor_internals import (
from torch.overrides import (
from torch.utils.dlpack import DLDeviceType
def rename_(self, *names, **rename_map):
    """In-place version of :meth:`~Tensor.rename`."""
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.rename_, (self,), self, *names, **rename_map)
    return update_names(self, names, rename_map, inplace=True)