import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
def is_evaluate_expr(self) -> bool:
    return self.name == 'evaluate_expr'