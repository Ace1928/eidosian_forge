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
def rhl1(self, alpha):
    """RHL1 Path."""
    self.name = 'RHL1'
    eta = (1 + 4 * cos(alpha)) / (2 + 4 * cos(alpha))
    nu = 3 / 4 - eta / 2.0
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'B': np.array([eta, 0.5, 1 - eta]), 'B_1': np.array([1 / 2.0, 1 - eta, eta - 1.0]), 'F': np.array([0.5, 0.5, 0.0]), 'L': np.array([0.5, 0.0, 0.0]), 'L_1': np.array([0.0, 0.0, -0.5]), 'P': np.array([eta, nu, nu]), 'P_1': np.array([1 - nu, 1 - nu, 1 - eta]), 'P_2': np.array([nu, nu, eta - 1.0]), 'Q': np.array([1 - nu, nu, 0.0]), 'X': np.array([nu, 0.0, -nu]), 'Z': np.array([0.5, 0.5, 0.5])}
    path = [['\\Gamma', 'L', 'B_1'], ['B', 'Z', '\\Gamma', 'X'], ['Q', 'F', 'P_1', 'Z'], ['L', 'P']]
    return {'kpoints': kpoints, 'path': path}