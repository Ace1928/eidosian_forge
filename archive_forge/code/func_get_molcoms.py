import numpy as np
import ase.units as units
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_masses
from ase.geometry import find_mic
def get_molcoms(self, nm):
    molcoms = np.zeros((nm, 3))
    for m in range(nm):
        molcoms[m] = self.atoms[m * 3:(m + 1) * 3].get_center_of_mass()
    return molcoms