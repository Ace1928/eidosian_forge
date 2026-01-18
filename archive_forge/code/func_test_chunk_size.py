import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
def test_chunk_size(chunk_size: int) -> bool:
    try:
        with torch.no_grad():
            fn(*args, chunk_size=chunk_size)
        return True
    except RuntimeError:
        return False