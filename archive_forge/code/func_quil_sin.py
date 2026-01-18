from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def quil_sin(expression: ExpressionDesignator) -> Function:
    return Function('SIN', expression, np.sin)