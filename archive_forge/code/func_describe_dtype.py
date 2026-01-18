from io import BytesIO
from itertools import product
import random
from typing import Any, List
import torch
def describe_dtype(dtype: torch.dtype) -> str:
    return DTYPE_NAMES.get(dtype) or str(dtype).rpartition('.')[2]