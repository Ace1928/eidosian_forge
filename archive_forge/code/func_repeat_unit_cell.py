from math import sqrt
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.gui.defaults import read_defaults
from ase.io import read, write, string2index
from ase.gui.i18n import _
from ase.geometry import find_mic
import warnings
def repeat_unit_cell(self):
    for atoms in self:
        results = self.repeat_results(atoms, self.repeat.prod(), oldprod=self.repeat.prod())
        atoms.cell *= self.repeat.reshape((3, 1))
        atoms.calc = SinglePointCalculator(atoms, **results)
    self.repeat = np.ones(3, int)