from typing import List, Optional
import numpy as np
from ase.data import atomic_numbers as ref_atomic_numbers
from ase.spacegroup import Spacegroup
from ase.cluster.base import ClusterBase
from ase.cluster.cluster import Cluster
def make_cluster(self, vacuum):
    size = np.array(self.size)
    translations = np.zeros((size.prod(), 3))
    for h in range(size[0]):
        for k in range(size[1]):
            for l in range(size[2]):
                i = h * (size[1] * size[2]) + k * size[2] + l
                translations[i] = np.dot([h, k, l], self.lattice_basis)
    atomic_basis = np.dot(self.atomic_basis, self.lattice_basis)
    positions = np.zeros((len(translations) * len(atomic_basis), 3))
    numbers = np.zeros(len(positions))
    n = len(atomic_basis)
    for i, trans in enumerate(translations):
        positions[n * i:n * (i + 1)] = atomic_basis + trans
        numbers[n * i:n * (i + 1)] = self.atomic_numbers
    for s, l in zip(self.surfaces, self.layers):
        n = self.miller_to_direction(s)
        rmax = self.get_layer_distance(s, l + 0.1)
        r = np.dot(positions - self.center, n)
        mask = np.less(r, rmax)
        if self.debug > 1:
            print('Cutting %s at %i layers ~ %.3f A' % (s, l, rmax))
        positions = positions[mask]
        numbers = numbers[mask]
    atoms = self.Cluster(symbols=numbers, positions=positions)
    atoms.cell = (1, 1, 1)
    atoms.center(about=(0, 0, 0))
    atoms.cell[:] = 0
    return atoms