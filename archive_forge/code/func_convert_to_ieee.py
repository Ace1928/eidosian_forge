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
def convert_to_ieee(self, structure: Structure, initial_fit=True, refine_rotation=True):
    """Convert all tensors to IEEE.

        Args:
            structure: Structure
            initial_fit: Whether to perform an initial fit.
            refine_rotation: Whether to refine the rotation.

        Returns:
            TensorCollection.
        """
    return type(self)([tensor.convert_to_ieee(structure, initial_fit, refine_rotation) for tensor in self])