from collections import Counter
import sys
import warnings
from time import time
from .apis import LpSolverDefault, PULP_CBC_CMD
from .apis.core import clock
from .utilities import value
from . import constants as const
from . import mps_lp as mpslp
import logging
import re
def findLHSValue(self):
    """
        for elastic constraints finds the LHS value of the constraint without
        the free variable and or penalty variable assumes the constant is on the
        rhs
        """
    if abs(value(self.denominator)) >= const.EPS:
        return value(self.numerator) / value(self.denominator)
    elif abs(value(self.numerator)) <= const.EPS:
        return 1.0
    else:
        raise ZeroDivisionError