import numpy as np
from collections import namedtuple
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality import topology_scaling
from ase.geometry.dimensionality.bond_generator import next_bond
def calculate_score(a, b):
    return f(b) - f(a)