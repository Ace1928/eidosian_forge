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
def asCplexVariablesOnly(self, name):
    """
        helper for asCplexLpAffineExpression
        """
    result = []
    line = [f'{name}:']
    notFirst = 0
    variables = self.sorted_keys()
    for v in variables:
        val = self[v]
        if val < 0:
            sign = ' -'
            val = -val
        elif notFirst:
            sign = ' +'
        else:
            sign = ''
        notFirst = 1
        if val == 1:
            term = f'{sign} {v.name}'
        else:
            term = f'{sign} {val + 0:.12g} {v.name}'
        if self._count_characters(line) + len(term) > const.LpCplexLPLineSize:
            result += [''.join(line)]
            line = [term]
        else:
            line += [term]
    return (result, line)