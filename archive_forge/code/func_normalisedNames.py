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
def normalisedNames(self):
    constraintsNames = {k: 'C%07d' % i for i, k in enumerate(self.constraints)}
    _variables = self.variables()
    variablesNames = {k.name: 'X%07d' % i for i, k in enumerate(_variables)}
    return (constraintsNames, variablesNames, 'OBJ')