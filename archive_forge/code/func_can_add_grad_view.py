from typing import Any, Callable, List, Optional, Union
import torch
def can_add_grad_view(self, param: torch.Tensor) -> bool:
    """Is there enough room in the bucket to add this parameter gradient, and is this param not already checked in ?"""
    return self._fill + param.numel() < self._max_size and id(param) not in self._param_ids