from random import randint
from typing import Dict, Tuple, Any
import numpy as np
from ase import Atoms
from ase.constraints import dict2constraint
from ase.calculators.calculator import (get_calculator_class, all_properties,
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols, atomic_masses
from ase.formula import Formula
from ase.geometry import cell_to_cellpar
from ase.io.jsonio import decode
def count_atoms(self):
    """Count atoms.

        Return dict mapping chemical symbol strings to number of atoms.
        """
    count = {}
    for symbol in self.symbols:
        count[symbol] = count.get(symbol, 0) + 1
    return count