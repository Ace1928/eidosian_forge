import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell
def get_mm_forces(self):
    """Calculate the forces on the MM-atoms from the QM-part."""
    f = self.pcpot.get_forces(self.qmatoms.calc)
    return self.mmatoms.calc.redistribute_forces(f)