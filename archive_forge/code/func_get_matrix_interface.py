import numbers
import numpy as np
import scipy.sparse as sp
from cvxpy.interface import numpy_interface as np_intf
def get_matrix_interface(target_class):
    return INTERFACES[target_class]