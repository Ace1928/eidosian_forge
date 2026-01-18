import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
def block2list(namespace, lines, header=None):
    """Parse lines of block and return list of lists of strings."""
    lines = iter(lines)
    block = []
    if header is None:
        header = next(lines)
    assert header.startswith('%'), header
    name = header[1:].strip().lower()
    for line in lines:
        if line.startswith('%'):
            break
        tokens = [namespace.evaluate(token) for token in line.strip().split('|')]
        block.append(tokens)
    return (name, block)