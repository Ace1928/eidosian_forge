import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
def get_positions_from_block(keyword):
    block = kwargs.pop(keyword)
    positions = []
    numbers = []
    tags = []
    types = {}
    for row in block:
        assert len(row) in [ndims + 1, ndims + 2]
        row = row[:ndims + 1]
        sym = row[0]
        assert sym.startswith('"') or sym.startswith("'")
        assert sym[0] == sym[-1]
        sym = sym[1:-1]
        pos0 = np.zeros(3)
        ndim = int(kwargs.get('dimensions', 3))
        pos0[:ndim] = [float(element) for element in row[1:]]
        number = atomic_numbers.get(sym)
        tag = 0
        if number is None:
            if sym not in types:
                tag = len(types) + 1
                types[sym] = tag
            number = 0
            tag = types[sym]
        tags.append(tag)
        numbers.append(number)
        positions.append(pos0)
    positions = np.array(positions)
    tags = np.array(tags, int)
    if types:
        ase_types = {}
        for sym, tag in types.items():
            ase_types['X', tag] = sym
        info = {'types': ase_types}
    else:
        tags = None
        info = None
    return (numbers, positions, tags, info)