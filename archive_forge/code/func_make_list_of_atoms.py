import math
from typing import Optional, Sequence
import numpy as np
from ase.atoms import Atoms
import ase.data
def make_list_of_atoms(self):
    """Repeat the unit cell."""
    nrep = self.size[0] * self.size[1] * self.size[2]
    if nrep <= 0:
        raise ValueError('Cannot create a non-positive number of unit cells')
    a2 = []
    e2 = []
    for i in range(self.size[0]):
        offset = self.basis[0] * i
        a2.append(self.atoms + offset[np.newaxis, :])
        e2.append(self.elements)
    atoms = np.concatenate(a2)
    elements = np.concatenate(e2)
    a2 = []
    e2 = []
    for j in range(self.size[1]):
        offset = self.basis[1] * j
        a2.append(atoms + offset[np.newaxis, :])
        e2.append(elements)
    atoms = np.concatenate(a2)
    elements = np.concatenate(e2)
    a2 = []
    e2 = []
    for k in range(self.size[2]):
        offset = self.basis[2] * k
        a2.append(atoms + offset[np.newaxis, :])
        e2.append(elements)
    atoms = np.concatenate(a2)
    elements = np.concatenate(e2)
    del a2, e2
    assert len(atoms) == nrep * len(self.atoms)
    basis = np.array([[self.size[0], 0, 0], [0, self.size[1], 0], [0, 0, self.size[2]]])
    basis = np.dot(basis, self.basis)
    basis = np.where(np.abs(basis) < self.chop_tolerance, 0.0, basis)
    lattice = Lattice(positions=atoms, cell=basis, numbers=elements, pbc=self.pbc)
    lattice.millerbasis = self.miller_basis
    lattice._addsorbate_info_size = np.array(self.size[:2])
    return lattice