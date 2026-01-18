from pathlib import Path
from re import compile
import numpy as np
from ase import Atoms
from ase.utils import reader
from ase.units import Bohr
def read_struct_out(fd):
    """Read a siesta struct file"""
    cell = []
    for i in range(3):
        line = next(fd)
        v = np.array(line.split(), float)
        cell.append(v)
    natoms = int(next(fd))
    numbers = np.empty(natoms, int)
    scaled_positions = np.empty((natoms, 3))
    for i, line in enumerate(fd):
        tokens = line.split()
        numbers[i] = int(tokens[1])
        scaled_positions[i] = np.array(tokens[2:5], float)
    return Atoms(numbers, cell=cell, pbc=True, scaled_positions=scaled_positions)