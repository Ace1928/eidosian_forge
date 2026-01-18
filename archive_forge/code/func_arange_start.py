import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def arange_start(start: number, end: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any):
    assert end >= 0
    assert end >= start
    return [int(math.ceil(end - start))]