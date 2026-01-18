import numpy as np
import ase.units as units
from ase.calculators.calculator import Calculator, all_changes
def add_virtual_sites(self, positions):
    return positions