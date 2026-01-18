from functools import reduce
from operator import mul
from typing import List, Tuple
def squeezed(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple((dim for dim in shape if dim != 1))