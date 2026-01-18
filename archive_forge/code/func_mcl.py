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
def mcl(self, b, c, beta):
    """MCL Path."""
    self.name = 'MCL'
    eta = (1 - b * cos(beta) / c) / (2 * sin(beta) ** 2)
    nu = 0.5 - eta * c * cos(beta) / b
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'A': np.array([0.5, 0.5, 0.0]), 'C': np.array([0.0, 0.5, 0.5]), 'D': np.array([0.5, 0.0, 0.5]), 'D_1': np.array([0.5, 0.5, -0.5]), 'E': np.array([0.5, 0.5, 0.5]), 'H': np.array([0.0, eta, 1 - nu]), 'H_1': np.array([0.0, 1 - eta, nu]), 'H_2': np.array([0.0, eta, -nu]), 'M': np.array([0.5, eta, 1 - nu]), 'M_1': np.array([0.5, 1 - eta, nu]), 'M_2': np.array([0.5, 1 - eta, nu]), 'X': np.array([0.0, 0.5, 0.0]), 'Y': np.array([0.0, 0.0, 0.5]), 'Y_1': np.array([0.0, 0.0, -0.5]), 'Z': np.array([0.5, 0.0, 0.0])}
    path = [['\\Gamma', 'Y', 'H', 'C', 'E', 'M_1', 'A', 'X', 'H_1'], ['M', 'D', 'Z'], ['Y', 'D']]
    return {'kpoints': kpoints, 'path': path}