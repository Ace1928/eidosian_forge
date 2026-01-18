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
def subInPlace(self, other):
    if isinstance(other, LpConstraint):
        if self.sense * other.sense <= 0:
            LpAffineExpression.subInPlace(self, other)
            self.sense |= -other.sense
        else:
            LpAffineExpression.addInPlace(self, other)
            self.sense |= other.sense
    elif isinstance(other, list):
        for e in other:
            self.subInPlace(e)
    else:
        LpAffineExpression.subInPlace(self, other)
    return self