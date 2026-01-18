import itertools
import numpy as np
from ase.utils import pbc2pbc
from ase.cell import Cell
def relevant_vectors_2D(u, v):
    cs = np.array([e for e in itertools.product([-1, 0, 1], repeat=2)])
    vs = cs @ [u, v]
    indices = np.argsort(np.linalg.norm(vs, axis=1))[:7]
    return (vs[indices], cs[indices])