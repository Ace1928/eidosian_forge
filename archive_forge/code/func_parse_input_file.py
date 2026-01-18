import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
def parse_input_file(fd):
    namespace = OctNamespace()
    lines = input_line_iter(fd)
    blocks = {}
    while True:
        try:
            line = next(lines)
        except StopIteration:
            break
        else:
            if line.startswith('%'):
                name, value = block2list(namespace, lines, header=line)
                blocks[name] = value
            else:
                tokens = line.split('=', 1)
                assert len(tokens) == 2, tokens
                name, value = tokens
                namespace.add(name, value)
    namespace.names.update(blocks)
    return namespace.names