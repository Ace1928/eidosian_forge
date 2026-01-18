from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def evaluate_objective(self):
    self._evaluate_objective_and_cache_if_necessary()
    return self._cached_objective