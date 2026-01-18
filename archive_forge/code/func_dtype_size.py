import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
def dtype_size(dtype):
    if dtype == torch.bool:
        return 1
    if dtype.is_floating_point or dtype.is_complex:
        return int(torch.finfo(dtype).bits / 8)
    return int(torch.iinfo(dtype).bits / 8)