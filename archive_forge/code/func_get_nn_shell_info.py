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
def get_nn_shell_info(self, structure: Structure, site_idx, shell):
    """Get a certain nearest neighbor shell for a certain site.

        Determines all non-backtracking paths through the neighbor network
        computed by `get_nn_info`. The weight is determined by multiplying
        the weight of the neighbor at each hop through the network. For
        example, a 2nd-nearest-neighbor that has a weight of 1 from its
        1st-nearest-neighbor and weight 0.5 from the original site will
        be assigned a weight of 0.5.

        As this calculation may involve computing the nearest neighbors of
        atoms multiple times, the calculation starts by computing all of the
        neighbor info and then calling `_get_nn_shell_info`. If you are likely
        to call this method for more than one site, consider calling `get_all_nn`
        first and then calling this protected method yourself.

        Args:
            structure (Molecule): Input structure
            site_idx (int): index of site for which to determine neighbor
                information.
            shell (int): Which neighbor shell to retrieve (1 == 1st NN shell)

        Returns:
            list of dictionaries. Each entry in the list is information about
                a certain neighbor in the structure, in the same format as
                `get_nn_info`.
        """
    all_nn_info = self.get_all_nn_info(structure)
    sites = self._get_nn_shell_info(structure, all_nn_info, site_idx, shell)
    output = []
    for info in sites:
        orig_site = structure[info['site_index']]
        info['site'] = Site(orig_site.species, orig_site._coords, properties=orig_site.properties)
        output.append(info)
    return output