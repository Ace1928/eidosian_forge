from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def random_orthogonal_matrix(dim, rng=np.random, real=False):
    """Generate a random orthogonal matrix"""
    H = rng.rand(dim, dim)
    np.add(dag(H), H, H)
    np.multiply(0.5, H, H)
    if real:
        gram_schmidt(H)
        return H
    else:
        val, vec = np.linalg.eig(H)
        return np.dot(vec * np.exp(1j * val), dag(vec))