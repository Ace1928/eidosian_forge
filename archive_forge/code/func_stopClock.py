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
def stopClock(self):
    """updates time wall time and cpu time"""
    self.solutionTime += time()
    self.solutionCpuTime += clock()