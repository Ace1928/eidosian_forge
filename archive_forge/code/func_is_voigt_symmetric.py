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
def is_voigt_symmetric(self, tol: float=1e-06) -> bool:
    """
        Args:
            tol: tolerance.

        Returns:
            Whether all tensors are voigt symmetric.
        """
    return all((tensor.is_voigt_symmetric(tol) for tensor in self))