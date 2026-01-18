import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def arange_start_step(start: number, end: number, step: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any):
    assert step != 0
    if step < 0:
        assert start >= end
    else:
        assert end >= start
    return [int(math.ceil((end - start) / step))]