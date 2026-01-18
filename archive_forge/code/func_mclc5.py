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
def mclc5(self, a, b, c, alpha):
    """MCLC5 Path."""
    self.name = 'MCLC5'
    zeta = (b ** 2 / a ** 2 + (1 - b * cos(alpha) / c) / sin(alpha) ** 2) / 4
    eta = 0.5 + 2 * zeta * c * cos(alpha) / b
    mu = eta / 2 + b ** 2 / (4 * a ** 2) - b * c * cos(alpha) / (2 * a ** 2)
    nu = 2 * mu - zeta
    rho = 1 - zeta * a ** 2 / b ** 2
    omega = (4 * nu - 1 - b ** 2 * sin(alpha) ** 2 / a ** 2) * c / (2 * b * cos(alpha))
    delta = zeta * c * cos(alpha) / b + omega / 2 - 0.25
    kpoints = {'\\Gamma': np.array([0.0, 0.0, 0.0]), 'F': np.array([nu, nu, omega]), 'F_1': np.array([1 - nu, 1 - nu, 1 - omega]), 'F_2': np.array([nu, nu - 1, omega]), 'H': np.array([zeta, zeta, eta]), 'H_1': np.array([1 - zeta, -zeta, 1 - eta]), 'H_2': np.array([-zeta, -zeta, 1 - eta]), 'I': np.array([rho, 1 - rho, 0.5]), 'I_1': np.array([1 - rho, rho - 1, 0.5]), 'L': np.array([0.5, 0.5, 0.5]), 'M': np.array([0.5, 0.0, 0.5]), 'N': np.array([0.5, 0.0, 0.0]), 'N_1': np.array([0.0, -0.5, 0.0]), 'X': np.array([0.5, -0.5, 0.0]), 'Y': np.array([mu, mu, delta]), 'Y_1': np.array([1 - mu, -mu, -delta]), 'Y_2': np.array([-mu, -mu, -delta]), 'Y_3': np.array([mu, mu - 1, delta]), 'Z': np.array([0.0, 0.0, 0.5])}
    path = [['\\Gamma', 'Y', 'F', 'L', 'I'], ['I_1', 'Z', 'H', 'F_1'], ['H_1', 'Y_1', 'X', '\\Gamma', 'N'], ['M', '\\Gamma']]
    return {'kpoints': kpoints, 'path': path}