import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def arange_end(end: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any):
    assert end >= 0
    return [int(math.ceil(end))]