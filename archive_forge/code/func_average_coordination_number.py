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
def average_coordination_number(structures, freq=10):
    """
    Calculates the ensemble averaged Voronoi coordination numbers
    of a list of Structures using VoronoiNN.
    Typically used for analyzing the output of a Molecular Dynamics run.

    Args:
        structures (list): list of Structures.
        freq (int): sampling frequency of coordination number [every freq steps].

    Returns:
        Dictionary of elements as keys and average coordination numbers as values.
    """
    coordination_numbers = {}
    for spec in structures[0].composition.as_dict():
        coordination_numbers[spec] = 0.0
    count = 0
    for idx, site in enumerate(structures):
        if idx % freq != 0:
            continue
        count += 1
        vnn = VoronoiNN()
        for j, atom in enumerate(site):
            cn = vnn.get_cn(site, j, use_weights=True)
            coordination_numbers[atom.species_string] += cn
    elements = structures[0].composition.as_dict()
    return {el: v / elements[el] / count for el, v in coordination_numbers.items()}