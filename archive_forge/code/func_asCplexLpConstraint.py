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
def asCplexLpConstraint(self, name):
    """
        Returns a constraint as a string
        """
    result, line = self.asCplexVariablesOnly(name)
    if not list(self.keys()):
        line += ['0']
    c = -self.constant
    if c == 0:
        c = 0
    term = f' {const.LpConstraintSenses[self.sense]} {c:.12g}'
    if self._count_characters(line) + len(term) > const.LpCplexLPLineSize:
        result += [''.join(line)]
        line = [term]
    else:
        line += [term]
    result += [''.join(line)]
    result = '%s\n' % '\n'.join(result)
    return result