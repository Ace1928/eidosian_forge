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
def set_dynamic(self, mask, value):
    for atoms in self:
        dynamic = self.get_dynamic(atoms)
        dynamic[mask[:len(atoms)]] = value
        atoms.constraints = [c for c in atoms.constraints if not isinstance(c, FixAtoms)]
        atoms.constraints.append(FixAtoms(mask=~dynamic))