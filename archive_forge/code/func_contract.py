import json
import numpy as np
from ase import Atoms
from ase.cell import Cell
def contract(dictionary):
    dcopy = {}
    for key in dictionary:
        dcopy[key.replace(' ', '').lower()] = dictionary[key]
    return dcopy