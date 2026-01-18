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
def nearest_key(sorted_vals: list[int], skey: int) -> int:
    n = bisect_left(sorted_vals, skey)
    if n == len(sorted_vals):
        return sorted_vals[-1]
    if n == 0:
        return sorted_vals[0]
    before = sorted_vals[n - 1]
    after = sorted_vals[n]
    if after - skey < skey - before:
        return after
    return before