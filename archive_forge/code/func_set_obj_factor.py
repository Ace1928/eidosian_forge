from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def set_obj_factor(self, obj_factor):
    self._invalidate_obj_factor_cache()
    self._obj_factor = obj_factor