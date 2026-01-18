from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def get_duals(self):
    return self._duals_full.copy()