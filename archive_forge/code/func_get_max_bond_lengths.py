from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_max_bond_lengths(structure, el_radius_updates=None):
    """
    Provides max bond length estimates for a structure based on the JMol
    table and algorithms.

    Args:
        structure: (structure)
        el_radius_updates: (dict) symbol->float to update atom_ic radii

    Returns:
        dict[(Element1, Element2)], float]: The two elements are ordered by Z.
    """
    jm_nn = JmolNN(el_radius_updates=el_radius_updates)
    bonds_lens = {}
    els = sorted(structure.elements, key=lambda x: x.Z)
    for i1, el1 in enumerate(els):
        for i2 in range(len(els) - i1):
            bonds_lens[el1, els[i1 + i2]] = jm_nn.get_max_bond_distance(el1.symbol, els[i1 + i2].symbol)
    return bonds_lens