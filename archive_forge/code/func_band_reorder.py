from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def band_reorder(self) -> None:
    """Re-order the eigenvalues according to the similarity of the eigenvectors."""
    eigen_displacements = self.eigendisplacements
    eig = self.bands
    n_phonons, n_qpoints = self.bands.shape
    order = np.zeros([n_qpoints, n_phonons], dtype=int)
    order[0] = np.array(range(n_phonons))
    assert self.structure is not None, 'Structure is required for band_reorder'
    atomic_masses = [site.specie.atomic_mass for site in self.structure]
    for nq in range(1, n_qpoints):
        old_eig_vecs = eigenvectors_from_displacements(eigen_displacements[:, nq - 1], atomic_masses)
        new_eig_vecs = eigenvectors_from_displacements(eigen_displacements[:, nq], atomic_masses)
        order[nq] = estimate_band_connection(old_eig_vecs.reshape([n_phonons, n_phonons]).T, new_eig_vecs.reshape([n_phonons, n_phonons]).T, order[nq - 1])
    for nq in range(1, n_qpoints):
        eivq = eigen_displacements[:, nq]
        eigq = eig[:, nq]
        eigen_displacements[:, nq] = eivq[order[nq]]
        eig[:, nq] = eigq[order[nq]]