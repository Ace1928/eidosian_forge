import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
def is_defer_runtime_assert(self) -> bool:
    return self.name == 'defer_runtime_assert'