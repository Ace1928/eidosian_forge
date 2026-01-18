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
def fixObjective(self):
    if self.objective is None:
        self.objective = 0
        wasNone = 1
    else:
        wasNone = 0
    if not isinstance(self.objective, LpAffineExpression):
        self.objective = LpAffineExpression(self.objective)
    if self.objective.isNumericalConstant():
        dummyVar = self.get_dummyVar()
        self.objective += dummyVar
    else:
        dummyVar = None
    return (wasNone, dummyVar)