from ase.units import Hartree, Bohr
from ase.io import write
import numpy as np
import os
from ase.calculators.calculator import FileIOCalculator
def manual_pc_correct(self):
    """ For current versions of CRYSTAL14/17, manual Coulomb correction """
    R = self.mmpositions / Bohr
    charges = self.mmcharges
    forces = np.zeros_like(R)
    energy = 0.0
    for m in range(len(charges)):
        D = R[m + 1:] - R[m]
        d2 = (D ** 2).sum(1)
        d = d2 ** 0.5
        e_c = charges[m + 1:] * charges[m] / d
        energy += np.sum(e_c)
        F = (e_c / d2)[:, None] * D
        forces[m] -= F.sum(0)
        forces[m + 1:] += F
    energy *= Hartree
    self.coulomb_corrections = (energy, forces)