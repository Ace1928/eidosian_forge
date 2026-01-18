from math import gcd
import re
from typing import Dict, Tuple, List, Sequence, Union
from ase.data import chemical_symbols, atomic_numbers
def formula_metal(numbers, empirical=False):
    """Convert list of atomic numbers to a chemical formula as a string.

    Elements are alphabetically ordered with metals first.

    If argument `empirical`, element counts will be divided by greatest common
    divisor to yield an empirical formula"""
    symbols = [chemical_symbols[Z] for Z in numbers]
    f = Formula('', _tree=[(symbols, 1)])
    if empirical:
        f, _ = f.reduce()
    return f.format('metal')