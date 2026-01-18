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
def get_okeeffe_distance_prediction(el1, el2):
    """
    Returns an estimate of the bond valence parameter (bond length) using
    the derived parameters from 'Atoms Sizes and Bond Lengths in Molecules
    and Crystals' (O'Keeffe & Brese, 1991). The estimate is based on two
    experimental parameters: r and c. The value for r  is based off radius,
    while c is (usually) the Allred-Rochow electronegativity. Values used
    are *not* generated from pymatgen, and are found in
    'okeeffe_params.json'.

    Args:
        el1, el2 (Element): two Element objects

    Returns:
        a float value of the predicted bond length
    """
    el1_okeeffe_params = get_okeeffe_params(el1)
    el2_okeeffe_params = get_okeeffe_params(el2)
    r1 = el1_okeeffe_params['r']
    r2 = el2_okeeffe_params['r']
    c1 = el1_okeeffe_params['c']
    c2 = el2_okeeffe_params['c']
    return r1 + r2 - r1 * r2 * pow(sqrt(c1) - sqrt(c2), 2) / (c1 * r1 + c2 * r2)