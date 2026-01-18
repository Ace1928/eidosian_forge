from __future__ import annotations
import collections
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
import scipy
from pymatgen.core.lattice import Lattice
from pymatgen.core.tensors import SquareTensor, symmetry_reduce
def apply_to_structure(self, structure: Structure):
    """
        Apply the deformation gradient to a structure.

        Args:
            structure (Structure object): the structure object to
                be modified by the deformation
        """
    def_struct = structure.copy()
    old_latt = def_struct.lattice.matrix
    new_latt = np.transpose(np.dot(self, np.transpose(old_latt)))
    def_struct.lattice = Lattice(new_latt)
    return def_struct