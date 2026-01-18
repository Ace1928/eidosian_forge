import math
from typing import List, Optional, Tuple
import numpy as np
import sympy
from cirq import ops, linalg, protocols
from cirq.linalg.tolerance import near_zero_mod
def is_quarter_turn(half_turns):
    return is_clifford_rotation(half_turns) and to_quarter_turns(half_turns) % 2 == 1