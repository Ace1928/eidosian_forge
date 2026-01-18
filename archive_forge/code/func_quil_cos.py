from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def quil_cos(expression: ExpressionDesignator) -> Function:
    return Function('COS', expression, np.cos)