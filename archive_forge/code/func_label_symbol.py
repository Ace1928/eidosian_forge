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
@staticmethod
def label_symbol(index):
    """Letters used in generating labels for the Latimer-Munro convention."""
    symbols = 'a b c d e f g h i j k l m n o p q r s t u v w x y z Γ'.split()
    return symbols[index]