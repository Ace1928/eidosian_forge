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
def isMIP(self):
    for v in self.variables():
        if v.cat == const.LpInteger:
            return 1
    return 0