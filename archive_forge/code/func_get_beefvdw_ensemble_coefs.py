import os
from typing import Any, Union
import numpy as np
from ase import Atoms
from ase.io.jsonio import read_json, write_json
from ase.parallel import world, parprint
def get_beefvdw_ensemble_coefs(self, size=2000, seed=0):
    """Perturbation coefficients of the BEEF-vdW ensemble"""
    from ase.dft.pars_beefvdw import uiOmega as omega
    assert np.shape(omega) == (31, 31)
    W, V, generator = self.eigendecomposition(omega, seed)
    RandV = generator.randn(31, size)
    for j in range(size):
        v = RandV[:, j]
        coefs_i = np.dot(np.dot(V, np.diag(np.sqrt(W))), v)[:]
        if j == 0:
            ensemble_coefs = coefs_i
        else:
            ensemble_coefs = np.vstack((ensemble_coefs, coefs_i))
    PBEc_ens = -ensemble_coefs[:, 30]
    return np.vstack((ensemble_coefs.T, PBEc_ens)).T