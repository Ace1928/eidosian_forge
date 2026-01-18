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
def assignConsSlack(self, values, activity=False):
    for name in values:
        try:
            if activity:
                self.constraints[name].slack = -1 * (self.constraints[name].constant + float(values[name]))
            else:
                self.constraints[name].slack = float(values[name])
        except KeyError:
            pass