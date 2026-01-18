from typing import (
import cmath
import re
import numpy as np
import sympy
def is_valid_end_state() -> bool:
    return len(vals) == 1 and len(ops) == 0