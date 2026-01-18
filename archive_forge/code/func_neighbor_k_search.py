from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
def neighbor_k_search(k_c, G_c, kpt_kc, tol=0.0001):
    alldir_dc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=int)
    for k0_c in alldir_dc:
        for k1, k1_c in enumerate(kpt_kc):
            if np.linalg.norm(k1_c - k_c - G_c + k0_c) < tol:
                return (k1, k0_c)
    print('Wannier: Did not find matching kpoint for kpt=', k_c)
    print('Probably non-uniform k-point grid')
    raise NotImplementedError