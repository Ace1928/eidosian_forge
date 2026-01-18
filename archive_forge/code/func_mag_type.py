from __future__ import annotations
import abc
import itertools
from math import ceil, cos, e, pi, sin, tan
from typing import TYPE_CHECKING, Any
from warnings import warn
import networkx as nx
import numpy as np
import spglib
from monty.dev import requires
from scipy.linalg import sqrtm
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, cite_conventional_cell_algo
@property
def mag_type(self):
    """
        Returns:
            The type of magnetic space group as a string. Current implementation does not
            distinguish between types 3 and 4, so return value is '3/4'. If has_magmoms is
            False, returns '0'.
        """
    return self._mag_type