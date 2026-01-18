import os
from typing import Any, Union
import numpy as np
from ase import Atoms
from ase.io.jsonio import read_json, write_json
from ase.parallel import world, parprint
def get_ensemble_energies(self, size: int=2000, seed: int=0) -> np.ndarray:
    """Returns an array of ensemble total energies"""
    self.seed = seed
    if self.verbose:
        parprint(self.beef_type, 'ensemble started')
    if self.contribs is None:
        self.contribs = self.calc.get_nonselfconsistent_energies(self.beef_type)
        self.e = self.calc.get_potential_energy(self.atoms)
    if self.beef_type == 'beefvdw':
        assert len(self.contribs) == 32
        coefs = self.get_beefvdw_ensemble_coefs(size, seed)
    elif self.beef_type == 'mbeef':
        assert len(self.contribs) == 64
        coefs = self.get_mbeef_ensemble_coefs(size, seed)
    elif self.beef_type == 'mbeefvdw':
        assert len(self.contribs) == 28
        coefs = self.get_mbeefvdw_ensemble_coefs(size, seed)
    self.de = np.dot(coefs, self.contribs)
    self.done = True
    if self.verbose:
        parprint(self.beef_type, 'ensemble finished')
    return self.e + self.de