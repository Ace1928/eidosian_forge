from __future__ import annotations
import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd, isclose
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, PeriodicSite, Structure, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.due import Doi, due
def build_slabgen_params() -> dict:
    """Build SlabGenerator parameters."""
    slabgen_params: dict = copy.deepcopy(recon_json['SlabGenerator_parameters'])
    slabgen_params['initial_structure'] = initial_structure.copy()
    slabgen_params['miller_index'] = recon_json['miller_index']
    slabgen_params['min_slab_size'] = min_slab_size
    slabgen_params['min_vacuum_size'] = min_vacuum_size
    return slabgen_params