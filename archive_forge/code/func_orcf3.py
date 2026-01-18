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
def orcf3(self, a, b, c):
    """ORFC3 Path."""
    self.name = 'ORCF3'
    zeta = (1 + a ** 2 / b ** 2 - a ** 2 / c ** 2) / 4
    eta = (1 + a ** 2 / b ** 2 + a ** 2 / c ** 2) / 4
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'A': np.array([0.5, 0.5 + zeta, zeta]), 'A_1': np.array([0.5, 0.5 - zeta, 1 - zeta]), 'L': np.array([0.5, 0.5, 0.5]), 'T': np.array([1, 0.5, 0.5]), 'X': np.array([0.0, eta, eta]), 'X_1': np.array([1, 1 - eta, 1 - eta]), 'Y': np.array([0.5, 0.0, 0.5]), 'Z': np.array([0.5, 0.5, 0.0])}
    path = [['\\Gamma', 'Y', 'T', 'Z', '\\Gamma', 'X', 'A_1', 'Y'], ['X', 'A', 'Z'], ['L', '\\Gamma']]
    return {'kpoints': kpoints, 'path': path}