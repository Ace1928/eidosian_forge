import pyomo.environ as pyo
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
def set_output_constraint_multipliers(self, output_con_multiplier_values):
    assert len(output_con_multiplier_values) == 1
    np.copyto(self._output_con_mult_values, output_con_multiplier_values)