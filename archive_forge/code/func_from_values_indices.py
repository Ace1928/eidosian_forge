from __future__ import annotations
import collections
import itertools
import os
import string
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from scipy.linalg import polar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@classmethod
def from_values_indices(cls, values, indices, populate=False, structure=None, voigt_rank=None, vsym=True, verbose=False) -> Self:
    """Creates a tensor from values and indices, with options
        for populating the remainder of the tensor.

        Args:
            values (floats): numbers to place at indices
            indices (array-likes): indices to place values at
            populate (bool): whether to populate the tensor
            structure (Structure): structure to base population
                or fit_to_structure on
            voigt_rank (int): full tensor rank to indicate the
                shape of the resulting tensor. This is necessary
                if one provides a set of indices more minimal than
                the shape of the tensor they want, e.g.
                Tensor.from_values_indices((0, 0), 100)
            vsym (bool): whether to voigt symmetrize during the
                optimization procedure
            verbose (bool): whether to populate verbosely
        """
    indices = np.array(indices)
    if voigt_rank:
        shape = np.array([3] * (voigt_rank % 2) + [6] * (voigt_rank // 2))
    else:
        shape = np.ceil(np.max(indices + 1, axis=0) / 3.0) * 3
    base = np.zeros(shape.astype(int))
    for v, idx in zip(values, indices):
        base[tuple(idx)] = v
    obj = cls.from_voigt(base) if 6 in shape else cls(base)
    if populate:
        assert structure, 'Populate option must include structure input'
        obj = obj.populate(structure, vsym=vsym, verbose=verbose)
    elif structure:
        obj = obj.fit_to_structure(structure)
    return obj