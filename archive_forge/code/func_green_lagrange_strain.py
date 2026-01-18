from __future__ import annotations
import collections
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
import scipy
from pymatgen.core.lattice import Lattice
from pymatgen.core.tensors import SquareTensor, symmetry_reduce
@property
def green_lagrange_strain(self):
    """Calculates the Euler-Lagrange strain from the deformation gradient."""
    return Strain.from_deformation(self)