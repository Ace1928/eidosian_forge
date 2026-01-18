import itertools
import warnings
from enum import auto, Enum
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, _get_param_to_fqns
from torch.distributed.fsdp._flat_param import FlatParamHandle
def get_handle_to_backward_prefetch(self, current_handle: FlatParamHandle) -> Optional[FlatParamHandle]:
    """
        Returns a :class:`list` of the handles keys of the handles to backward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
    current_index = current_handle._post_forward_index
    if current_index is None:
        return None
    target_index = current_index - 1
    target_handle: Optional[FlatParamHandle] = None
    for _ in range(self._backward_prefetch_limit):
        if target_index < 0:
            break
        target_handle = self.handles_post_forward_order[target_index]
        target_index -= 1
    return target_handle