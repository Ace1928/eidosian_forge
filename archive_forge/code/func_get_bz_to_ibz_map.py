import numpy as np
from ase.calculators.calculator import Calculator, all_properties
from ase.calculators.calculator import PropertyNotImplementedError
def get_bz_to_ibz_map(self):
    return self.bz2ibz