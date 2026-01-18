from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def quil_exp(expression: ExpressionDesignator) -> Function:
    return Function('EXP', expression, np.exp)