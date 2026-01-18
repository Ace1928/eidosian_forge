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
def checkDuplicateVars(self) -> None:
    """
        Checks if there are at least two variables with the same name
        :return: 1
        :raises `const.PulpError`: if there ar duplicates
        """
    name_counter = Counter((variable.name for variable in self.variables()))
    repeated_names = {(name, count) for name, count in name_counter.items() if count >= 2}
    if repeated_names:
        raise const.PulpError(f'Repeated variable names: {repeated_names}')