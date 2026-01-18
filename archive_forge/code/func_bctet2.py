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
def bctet2(self, c, a):
    """BCT2 Path."""
    self.name = 'BCT2'
    eta = (1 + a ** 2 / c ** 2) / 4.0
    zeta = a ** 2 / (2 * c ** 2)
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'N': np.array([0.0, 0.5, 0.0]), 'P': np.array([0.25, 0.25, 0.25]), '\\Sigma': np.array([-eta, eta, eta]), '\\Sigma_1': np.array([eta, 1 - eta, -eta]), 'X': np.array([0.0, 0.0, 0.5]), 'Y': np.array([-zeta, zeta, 0.5]), 'Y_1': np.array([0.5, 0.5, -zeta]), 'Z': np.array([0.5, 0.5, -0.5])}
    path = [['\\Gamma', 'X', 'Y', '\\Sigma', '\\Gamma', 'Z', '\\Sigma_1', 'N', 'P', 'Y_1', 'Z'], ['X', 'P']]
    return {'kpoints': kpoints, 'path': path}