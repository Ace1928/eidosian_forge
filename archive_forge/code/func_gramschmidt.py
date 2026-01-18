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
def gramschmidt(vin, uin):
    """
    Returns that part of the first input vector
    that is orthogonal to the second input vector.
    The output vector is not normalized.

    Args:
        vin (numpy array):
            first input vector
        uin (numpy array):
            second input vector
    """
    vin_uin = np.inner(vin, uin)
    uin_uin = np.inner(uin, uin)
    if uin_uin <= 0.0:
        raise ValueError('Zero or negative inner product!')
    return vin - vin_uin / uin_uin * uin