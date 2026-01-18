import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
def storage_offset(self) -> Union[int, torch.SymInt]:
    return self.tensor_storage_offset