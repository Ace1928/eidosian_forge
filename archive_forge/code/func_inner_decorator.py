from builtins import isinstance
import functools
import logging
from typing import Any, List, Tuple
import torch
from torch import nn
@functools.wraps(orig_func)
def inner_decorator(*args: Any, **kwargs: Any) -> Any:
    return orig_func(*args, **kwargs)