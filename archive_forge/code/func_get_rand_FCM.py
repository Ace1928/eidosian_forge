from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@requires(Phonopy, 'phonopy not installed!')
def get_rand_FCM(self, asum=15, force=10):
    """
        Generate a symmetrized force constant matrix from an unsymmetrized matrix
        that has no unstable modes and also obeys the acoustic sum rule through an
        iterative procedure.

        Args:
            force (float): maximum force constant
            asum (int): number of iterations to attempt to obey the acoustic sum
                rule

        Returns:
            NxNx3x3 np.array representing the force constant matrix
        """
    from pymatgen.io.phonopy import get_phonopy_structure
    n_sites = len(self.structure)
    structure = get_phonopy_structure(self.structure)
    pn_struct = Phonopy(structure, np.eye(3), np.eye(3))
    dyn = self.get_unstable_FCM(force)
    dyn = self.get_stable_FCM(dyn)
    dyn = np.reshape(dyn, (n_sites, 3, n_sites, 3)).swapaxes(1, 2)
    dyn_mass = np.zeros([len(self.structure), len(self.structure), 3, 3])
    masses = []
    for idx in range(n_sites):
        masses.append(self.structure[idx].specie.atomic_mass)
    dyn_mass = np.zeros([n_sites, n_sites, 3, 3])
    for m in range(n_sites):
        for n in range(n_sites):
            dyn_mass[m][n] = dyn[m][n] * np.sqrt(masses[m]) * np.sqrt(masses[n])
    supercell = pn_struct.get_supercell()
    primitive = pn_struct.get_primitive()
    converter = dyntofc.DynmatToForceConstants(primitive, supercell)
    dyn = np.reshape(np.swapaxes(dyn_mass, 1, 2), (n_sites * 3, n_sites * 3))
    converter.set_dynamical_matrices(dynmat=[dyn])
    converter.run()
    return converter.get_force_constants()