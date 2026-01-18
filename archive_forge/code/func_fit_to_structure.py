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
def fit_to_structure(self, structure: Structure, symprec: float=0.1):
    """Fits all tensors to a Structure.

        Args:
            structure: Structure
            symprec: symmetry precision.

        Returns:
            TensorCollection.
        """
    return type(self)([tensor.fit_to_structure(structure, symprec) for tensor in self])