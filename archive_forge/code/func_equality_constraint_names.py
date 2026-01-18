import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
import scipy.sparse as spa
import numpy as np
import math
def equality_constraint_names(self):
    return self._eq_constraint_names