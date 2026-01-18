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
def isBinary(self):
    return self.cat == const.LpInteger and self.lowBound == 0 and (self.upBound == 1)