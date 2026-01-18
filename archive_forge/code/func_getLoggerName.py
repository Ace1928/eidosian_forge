from .base_linear_solver_interface import IPLinearSolverInterface
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus, LinearSolverResults
from pyomo.common.dependencies import attempt_import
from collections import OrderedDict
from typing import Union, Optional, Tuple
from pyomo.contrib.pynumero.sparse import BlockVector
import numpy as np
from pyomo.contrib.pynumero.linalg.mumps_interface import (
@classmethod
def getLoggerName(cls):
    return 'mumps'