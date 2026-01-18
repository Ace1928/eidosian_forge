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
def get_symbol_dict(self, voigt=True, zero_index=False, **kwargs):
    """Creates a summary dict for tensor with associated symbol.

        Args:
            voigt (bool): whether to get symbol dict for voigt
                notation tensor, as opposed to full notation,
                defaults to true
            zero_index (bool): whether to set initial index to zero,
                defaults to false, since tensor notations tend to use
                one-indexing, rather than zero indexing like python
            **kwargs: keyword args for np.isclose. Can take atol
                and rtol for absolute and relative tolerance, e. g.

                >>> tensor.get_symbol_dict(atol=1e-8)

                or

                >>> tensor.get_symbol_dict(rtol=1e-5)

        Returns:
            list of index groups where tensor values are equivalent to
            within tolerances
        """
    dct = {}
    array = self.voigt if voigt else self
    grouped = self.get_grouped_indices(voigt=voigt, **kwargs)
    p = 0 if zero_index else 1
    for indices in grouped:
        sym_string = self.symbol + '_'
        sym_string += ''.join((str(i + p) for i in indices[0]))
        value = array[indices[0]]
        if not np.isclose(value, 0):
            dct[sym_string] = array[indices[0]]
    return dct