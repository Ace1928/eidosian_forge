import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.build import molecule
from ase.ga.utilities import (closest_distances_generator, atoms_too_close,
def generate_unit_cell(self, repeat):
    """Generates a random unit cell.

        For this, we use the vectors in self.slab.cell
        in the fixed directions and randomly generate
        the variable ones. For such a cell to be valid,
        it has to satisfy the self.cellbounds constraints.

        The cell will also be such that the volume of the
        box in which the atoms can be placed (box limits
        described by self.box_to_place_in) is equal to
        self.box_volume.

        Parameters:

        repeat: tuple of 3 integers
            Indicates by how much each cell vector
            will later be reduced by cell splitting.

            This is used to ensure that the original
            cell is large enough so that the cell lengths
            of the smaller cell exceed the largest
            (X,X)-minimal-interatomic-distance in self.blmin.
        """
    Lmin = 0.0
    for atoms, count in self.blocks:
        dist = atoms.get_all_distances(mic=False, vector=False)
        num = atoms.get_atomic_numbers()
        for i in range(len(atoms)):
            dist[i, i] += self.blmin[num[i], num[i]]
            for j in range(i):
                bl = self.blmin[num[i], num[j]]
                dist[i, j] += bl
                dist[j, i] += bl
        L = np.max(dist)
        if L > Lmin:
            Lmin = L
    valid = False
    while not valid:
        cell = np.zeros((3, 3))
        for i in range(self.number_of_variable_cell_vectors):
            cell[i, i] = self.rng.rand() * np.cbrt(self.box_volume)
            cell[i, i] *= repeat[i]
            for j in range(i):
                cell[i, j] = (self.rng.rand() - 0.5) * cell[i - 1, i - 1]
        for i in range(self.number_of_variable_cell_vectors, 3):
            cell[i] = self.box_to_place_in[1][i]
        if self.number_of_variable_cell_vectors > 0:
            volume = abs(np.linalg.det(cell))
            scaling = self.box_volume / volume
            scaling **= 1.0 / self.number_of_variable_cell_vectors
            cell[:self.number_of_variable_cell_vectors] *= scaling
        for i in range(self.number_of_variable_cell_vectors, 3):
            cell[i] = self.slab.get_cell()[i]
        valid = True
        if self.cellbounds is not None:
            if not self.cellbounds.is_within_bounds(cell):
                valid = False
        if valid:
            for i in range(3):
                if np.linalg.norm(cell[i]) < repeat[i] * Lmin:
                    assert self.number_of_variable_cell_vectors > 0
                    valid = False
    return cell