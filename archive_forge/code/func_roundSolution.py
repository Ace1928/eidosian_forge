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
def roundSolution(self, epsInt=1e-05, eps=1e-07):
    """
        Rounds the lp variables

        Inputs:
            - none

        Side Effects:
            - The lp variables are rounded
        """
    for v in self.variables():
        v.round(epsInt, eps)