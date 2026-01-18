import os
import numpy as np
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError, Parameters
from ase.units import kcal, mol, Debye
def get_final_heat_of_formation(self):
    """Final heat of formation as reported in the Mopac output file
        """
    return self.final_hof