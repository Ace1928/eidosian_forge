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
def average_over_unit_sphere(self, quad=None):
    """Method for averaging the tensor projection over the unit
        with option for custom quadrature.

        Args:
            quad (dict): quadrature for integration, should be
                dictionary with "points" and "weights" keys defaults
                to quadpy.sphere.Lebedev(19) as read from file

        Returns:
            Average of tensor projected into vectors on the unit sphere
        """
    quad = quad or DEFAULT_QUAD
    weights, points = (quad['weights'], quad['points'])
    return sum((w * self.project(n) for w, n in zip(weights, points)))