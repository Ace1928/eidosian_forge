import collections
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.utils import reader, writer
def match_int(line, word):
    number, colon, word1 = line.split()
    assert word1 == word
    assert colon == ':'
    return int(number)