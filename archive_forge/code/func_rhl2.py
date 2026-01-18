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
def rhl2(self, alpha):
    """RHL2 Path."""
    self.name = 'RHL2'
    eta = 1 / (2 * tan(alpha / 2.0) ** 2)
    nu = 3 / 4 - eta / 2.0
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'F': np.array([0.5, -0.5, 0.0]), 'L': np.array([0.5, 0.0, 0.0]), 'P': np.array([1 - nu, -nu, 1 - nu]), 'P_1': np.array([nu, nu - 1.0, nu - 1.0]), 'Q': np.array([eta, eta, eta]), 'Q_1': np.array([1 - eta, -eta, -eta]), 'Z': np.array([0.5, -0.5, 0.5])}
    path = [['\\Gamma', 'P', 'Z', 'Q', '\\Gamma', 'F', 'P_1', 'Q_1', 'L', 'Z']]
    return {'kpoints': kpoints, 'path': path}