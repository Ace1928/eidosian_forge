from __future__ import annotations
import collections
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
import scipy
from pymatgen.core.lattice import Lattice
from pymatgen.core.tensors import SquareTensor, symmetry_reduce
def get_deformation_matrix(self, shape: Literal['upper', 'lower', 'symmetric']='upper'):
    """
        Returns the deformation matrix.

        Args:
            shape ('upper' | 'lower' | 'symmetric'): method for determining deformation
                'upper' produces an upper triangular defo
                'lower' produces a lower triangular defo
                'symmetric' produces a symmetric defo
        """
    return convert_strain_to_deformation(self, shape=shape)