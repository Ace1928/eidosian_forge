import numpy as np
from ase.io.jsonio import read_json, write_json
def calculate_ldos(self, bias):
    """Calculate local density of states for given bias."""
    if self.ldos is not None and bias == self.bias:
        return
    self.bias = bias
    calc = self.atoms.calc
    if self.use_density:
        self.ldos = calc.get_pseudo_density()
        return
    if bias < 0:
        emin = bias
        emax = 0.0
    else:
        emin = 0
        emax = bias
    nbands = calc.get_number_of_bands()
    weights = calc.get_k_point_weights()
    nkpts = len(weights)
    nspins = calc.get_number_of_spins()
    eigs = np.array([[calc.get_eigenvalues(k, s) for k in range(nkpts)] for s in range(nspins)])
    eigs -= calc.get_fermi_level()
    ldos = np.zeros(calc.get_pseudo_wave_function(0, 0, 0).shape)
    for s in range(nspins):
        for k in range(nkpts):
            for n in range(nbands):
                e = eigs[s, k, n]
                if emin < e < emax:
                    psi = calc.get_pseudo_wave_function(n, k, s)
                    ldos += weights[k] * (psi * np.conj(psi)).real
    if 0 in self.symmetries:
        ldos[1:] += ldos[:0:-1].copy()
        ldos[1:] *= 0.5
    if 1 in self.symmetries:
        ldos[:, 1:] += ldos[:, :0:-1].copy()
        ldos[:, 1:] *= 0.5
    if 2 in self.symmetries:
        ldos += ldos.transpose((1, 0, 2)).copy()
        ldos *= 0.5
    self.ldos = ldos