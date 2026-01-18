import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
def is_create_fx_call_function(self) -> bool:
    return self.name == 'create_fx_call_function'