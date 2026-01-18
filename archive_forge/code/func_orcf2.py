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
def orcf2(self, a, b, c):
    """ORFC2 Path."""
    self.name = 'ORCF2'
    phi = (1 + c ** 2 / b ** 2 - c ** 2 / a ** 2) / 4
    eta = (1 + a ** 2 / b ** 2 - a ** 2 / c ** 2) / 4
    delta = (1 + b ** 2 / a ** 2 - b ** 2 / c ** 2) / 4
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'C': np.array([0.5, 0.5 - eta, 1 - eta]), 'C_1': np.array([0.5, 0.5 + eta, eta]), 'D': np.array([0.5 - delta, 0.5, 1 - delta]), 'D_1': np.array([0.5 + delta, 0.5, delta]), 'L': np.array([0.5, 0.5, 0.5]), 'H': np.array([1 - phi, 0.5 - phi, 0.5]), 'H_1': np.array([phi, 0.5 + phi, 0.5]), 'X': np.array([0.0, 0.5, 0.5]), 'Y': np.array([0.5, 0.0, 0.5]), 'Z': np.array([0.5, 0.5, 0.0])}
    path = [['\\Gamma', 'Y', 'C', 'D', 'X', '\\Gamma', 'Z', 'D_1', 'H', 'C'], ['C_1', 'Z'], ['X', 'H_1'], ['H', 'Y'], ['L', '\\Gamma']]
    return {'kpoints': kpoints, 'path': path}