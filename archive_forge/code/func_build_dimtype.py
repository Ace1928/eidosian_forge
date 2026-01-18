import numpy as np
from collections import namedtuple
from ase.geometry.dimensionality import rank_determination
from ase.geometry.dimensionality import topology_scaling
from ase.geometry.dimensionality.bond_generator import next_bond
def build_dimtype(h):
    h = reduced_histogram(h)
    return ''.join([str(i) for i, e in enumerate(h) if e > 0]) + 'D'