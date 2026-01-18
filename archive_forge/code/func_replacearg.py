import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
def replacearg(index: int, key: str, fn: Callable):
    if index < len(args):
        args[index] = fn(args[index])
    if key in kwargs:
        kwargs[key] = fn(kwargs[key])