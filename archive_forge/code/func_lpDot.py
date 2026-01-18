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
def lpDot(v1, v2):
    """Calculate the dot product of two lists of linear expressions"""
    if not const.isiterable(v1) and (not const.isiterable(v2)):
        return v1 * v2
    elif not const.isiterable(v1):
        return lpDot([v1] * len(v2), v2)
    elif not const.isiterable(v2):
        return lpDot(v1, [v2] * len(v1))
    else:
        return lpSum([lpDot(e1, e2) for e1, e2 in zip(v1, v2)])