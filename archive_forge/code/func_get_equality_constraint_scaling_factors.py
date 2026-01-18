import pyomo.environ as pyo
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
def get_equality_constraint_scaling_factors(self):
    return np.asarray([0.1, 0.2, 0.3, 0.4])