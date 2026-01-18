import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def conv_backwards(grad_output: List[int], input: List[int], weight: List[int], biases: Optional[List[int]]):
    return (_copy(input), _copy(weight), [grad_output[1]])