import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def calculate_forces(self, atoms):
    self.update(atoms)
    self.results['forces'] = np.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        neighbors, offsets = self.neighbors.get_neighbors(i)
        offset = np.dot(offsets, atoms.get_cell())
        rvec = atoms.positions[neighbors] + offset - atoms.positions[i]
        r = np.sqrt(np.sum(np.square(rvec), axis=1))
        nearest = np.arange(len(r))[r < self.cutoff]
        d_embedded_energy_i = self.d_embedded_energy[self.index[i]](self.total_density[i])
        urvec = rvec.copy()
        for j in np.arange(len(neighbors)):
            urvec[j] = urvec[j] / r[j]
        for j_index in range(self.Nelements):
            use = self.index[neighbors[nearest]] == j_index
            if not use.any():
                continue
            rnuse = r[nearest][use]
            density_j = self.total_density[neighbors[nearest][use]]
            if self.form == 'fs':
                scale = self.d_phi[self.index[i], j_index](rnuse) + d_embedded_energy_i * self.d_electron_density[j_index, self.index[i]](rnuse) + self.d_embedded_energy[j_index](density_j) * self.d_electron_density[self.index[i], j_index](rnuse)
            else:
                scale = self.d_phi[self.index[i], j_index](rnuse) + d_embedded_energy_i * self.d_electron_density[j_index](rnuse) + self.d_embedded_energy[j_index](density_j) * self.d_electron_density[self.index[i]](rnuse)
            self.results['forces'][i] += np.dot(scale, urvec[nearest][use])
            if self.form == 'adp':
                adp_forces = self.angular_forces(self.mu[i], self.mu[neighbors[nearest][use]], self.lam[i], self.lam[neighbors[nearest][use]], rnuse, rvec[nearest][use], self.index[i], j_index)
                self.results['forces'][i] += adp_forces