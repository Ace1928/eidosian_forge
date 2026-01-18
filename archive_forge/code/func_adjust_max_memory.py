import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Union
from packaging import version
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
    max_memory = {key: val * 0.9 for key, val in max_memory.items()}
    return max_memory