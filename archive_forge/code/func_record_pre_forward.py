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
def record_pre_forward(self, handle: Optional[FlatParamHandle], is_training: bool) -> None:
    """
        Records ``handles`` in the pre-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        On the first iteration, this checks the execution order across ranks.
        See :meth:`_check_order` for details.
        """
    if not handle:
        return
    self._check_order(handle, is_training)
    if not self.is_first_iter or handle._pre_forward_order_index is not None:
        return
    index = len(self.handles_pre_forward_order)
    handle._pre_forward_order_index = index
    self.handles_pre_forward_order.append(handle)