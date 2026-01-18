import numpy as np
from .structreader import Unpacker
from .utils import find_private_section
def get_b_matrix(csa_dict):
    vals = get_vector(csa_dict, 'B_matrix', 6)
    if vals is None:
        return
    inds = np.array([0, 1, 2, 1, 3, 4, 2, 4, 5])
    B = np.array(vals)[inds]
    return B.reshape(3, 3)