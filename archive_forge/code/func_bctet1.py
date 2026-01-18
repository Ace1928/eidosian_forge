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
def bctet1(self, c, a):
    """BCT1 Path."""
    self.name = 'BCT1'
    eta = (1 + c ** 2 / a ** 2) / 4.0
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'M': np.array([-0.5, 0.5, 0.5]), 'N': np.array([0.0, 0.5, 0.0]), 'P': np.array([0.25, 0.25, 0.25]), 'X': np.array([0.0, 0.0, 0.5]), 'Z': np.array([eta, eta, -eta]), 'Z_1': np.array([-eta, 1 - eta, eta])}
    path = [['\\Gamma', 'X', 'M', '\\Gamma', 'Z', 'P', 'N', 'Z_1', 'M'], ['X', 'P']]
    return {'kpoints': kpoints, 'path': path}