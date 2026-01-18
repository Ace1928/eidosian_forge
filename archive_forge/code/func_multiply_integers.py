import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def multiply_integers(li: List[int]):
    out = 1
    for elem in li:
        out = out * elem
    return out