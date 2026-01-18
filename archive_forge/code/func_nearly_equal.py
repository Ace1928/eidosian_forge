import numpy as np
from ase.geometry import cell_to_cellpar as c2p, cellpar_to_cell as p2c
def nearly_equal(a, b):
    return np.all(np.abs(b - a) < eps)