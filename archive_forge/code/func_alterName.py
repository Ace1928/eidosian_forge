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
def alterName(self, name):
    """
        Alters the name of anonymous parts of the problem

        """
    self.name = f'{name}_elastic_SubProblem'
    if hasattr(self, 'freeVar'):
        self.freeVar.name = self.name + '_free_bound'
    if hasattr(self, 'upVar'):
        self.upVar.name = self.name + '_pos_penalty_var'
    if hasattr(self, 'lowVar'):
        self.lowVar.name = self.name + '_neg_penalty_var'