from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def same_half_turns(a1: float, a2: float, atol=0.0001) -> bool:
    d = (a1 - a2 + 1) % 2 - 1
    return abs(d) < atol