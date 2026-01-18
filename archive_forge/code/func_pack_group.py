import math
import functools
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain
from typing import (
from typing_extensions import ParamSpec, Self, TypeAlias
import torch
import torch.utils.hooks as hooks
from torch.utils.hooks import RemovableHandle
from torch.utils._foreach_utils import (
from torch._utils import is_compiling
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
def pack_group(group: Dict[str, Any]) -> Dict[str, Any]:
    nonlocal start_index
    packed = {k: v for k, v in group.items() if k != 'params'}
    param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index) if id(p) not in param_mappings})
    packed['params'] = [param_mappings[id(p)] for p in group['params']]
    start_index += len(packed['params'])
    return packed