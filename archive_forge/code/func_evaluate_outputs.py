import numpy as np
from scipy.optimize import fsolve
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
def evaluate_outputs(self):
    sv = self._input_values[0]
    caf = self._input_values[1]
    k1 = self._input_values[2]
    k2 = self._input_values[3]
    k3 = self._input_values[4]
    ret = reactor_outlet_concentrations(sv, caf, k1, k2, k3)
    return np.asarray(ret, dtype=np.float64)