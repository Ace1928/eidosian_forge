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
def setObjective(self, obj):
    """
        Sets the input variable as the objective function. Used in Columnwise Modelling

        :param obj: the objective function of type :class:`LpConstraintVar`

        Side Effects:
            - The objective function is set
        """
    if isinstance(obj, LpVariable):
        obj = obj + 0.0
    try:
        obj = obj.constraint
        name = obj.name
    except AttributeError:
        name = None
    self.objective = obj
    self.objective.name = name
    self.resolveOK = False