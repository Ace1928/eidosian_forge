import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def count_symbols(atoms, exclude=()):
    """Count symbols in atoms object, excluding a set of indices
    
    Parameters:
        atoms: Atoms object to be grouped
        exclude: List of indices to be excluded from the counting
    
    Returns:
        Tuple of (symbols, symbolcount)
        symbols: The unique symbols in the included list
        symbolscount: Count of symbols in the included list

    Example:

    >>> from ase.build import bulk
    >>> atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.1, cubic=True)
    >>> count_symbols(atoms)
    (['Na', 'Cl'], {'Na': 4, 'Cl': 4})
    >>> count_symbols(atoms, exclude=(1, 2, 3))
    (['Na', 'Cl'], {'Na': 3, 'Cl': 2})
    """
    symbols = []
    symbolcount = {}
    for m, symbol in enumerate(atoms.symbols):
        if m in exclude:
            continue
        if symbol not in symbols:
            symbols.append(symbol)
            symbolcount[symbol] = 1
        else:
            symbolcount[symbol] += 1
    return (symbols, symbolcount)