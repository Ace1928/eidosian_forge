from __future__ import annotations
import json
import math
import os
import warnings
from bisect import bisect_left
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from math import acos, asin, atan2, cos, exp, fabs, pi, pow, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal, get_args
import numpy as np
from monty.dev import deprecated, requires
from monty.serialization import loadfn
from ruamel.yaml import YAML
from scipy.spatial import Voronoi
from pymatgen.analysis.bond_valence import BV_PARAMS, BVAnalyzer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, IStructure, PeriodicNeighbor, PeriodicSite, Site, Species, Structure
def get_cn(self, structure: Structure, n: int, **kwargs) -> float:
    """
        Get coordination number, CN, of site with index n in structure.

        Args:
            structure (Structure): input structure.
            n (int): index of site for which to determine CN.
            use_weights (bool): flag indicating whether (True)
                to use weights for computing the coordination number
                or not (False, default: each coordinated site has equal
                weight).
            on_disorder ('take_majority_strict' | 'take_majority_drop' | 'take_max_species' | 'error'):
                What to do when encountering a disordered structure. 'error' will raise ValueError.
                'take_majority_strict' will use the majority specie on each site and raise
                ValueError if no majority exists. 'take_max_species' will use the first max specie
                on each site. For {{Fe: 0.4, O: 0.4, C: 0.2}}, 'error' and 'take_majority_strict'
                will raise ValueError, while 'take_majority_drop' ignores this site altogether and
                'take_max_species' will use Fe as the site specie.

        Returns:
            cn (float): coordination number.
        """
    use_weights = kwargs.get('use_weights', False)
    if self.weighted_cn != use_weights:
        raise ValueError('The weighted_cn parameter and use_weights parameter should match!')
    return super().get_cn(structure, n, **kwargs)