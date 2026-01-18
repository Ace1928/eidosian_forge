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
def bcc(self):
    """BCC Path."""
    self.name = 'BCC'
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'H': np.array([0.5, -0.5, 0.5]), 'P': np.array([0.25, 0.25, 0.25]), 'N': np.array([0.0, 0.0, 0.5])}
    path = [['\\Gamma', 'H', 'N', '\\Gamma', 'P', 'H'], ['P', 'N']]
    return {'kpoints': kpoints, 'path': path}