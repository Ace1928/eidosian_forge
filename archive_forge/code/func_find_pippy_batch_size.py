import math
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union
from .state import PartialState
from .utils import (
def find_pippy_batch_size(args, kwargs):
    found_batch_size = None
    if args is not None:
        for arg in args:
            found_batch_size = ignorant_find_batch_size(arg)
            if found_batch_size is not None:
                break
    if kwargs is not None and found_batch_size is None:
        for kwarg in kwargs.values():
            found_batch_size = ignorant_find_batch_size(kwarg)
            if found_batch_size is not None:
                break
    return found_batch_size