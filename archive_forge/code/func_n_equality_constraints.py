import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
import scipy.sparse as spa
import numpy as np
import math
def n_equality_constraints(self):
    return len(self.equality_constraint_names())