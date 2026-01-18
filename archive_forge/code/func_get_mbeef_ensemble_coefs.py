import os
from typing import Any, Union
import numpy as np
from ase import Atoms
from ase.io.jsonio import read_json, write_json
from ase.parallel import world, parprint
def get_mbeef_ensemble_coefs(self, size=2000, seed=0):
    """Perturbation coefficients of the mBEEF ensemble"""
    from ase.dft.pars_mbeef import uiOmega as omega
    assert np.shape(omega) == (64, 64)
    W, V, generator = self.eigendecomposition(omega, seed)
    mu, sigma = (0.0, 1.0)
    rand = np.array(generator.normal(mu, sigma, (len(W), size)))
    return (np.sqrt(2) * np.dot(np.dot(V, np.diag(np.sqrt(W))), rand)[:]).T