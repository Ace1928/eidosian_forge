import threading
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from ..modules import Module
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper
def get_a_var(obj: Union[torch.Tensor, List[Any], Tuple[Any, ...], Dict[Any, Any]]) -> Optional[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None