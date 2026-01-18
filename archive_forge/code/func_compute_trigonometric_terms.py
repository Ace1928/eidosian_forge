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
def compute_trigonometric_terms(self, thetas, phis):
    """
        Computes trigonometric terms that are required to
        calculate bond orientational order parameters using
        internal variables.

        Args:
            thetas ([float]): polar angles of all neighbors in radians.
            phis ([float]): azimuth angles of all neighbors in radians.
                The list of
                azimuth angles of all neighbors in radians. The list of
                azimuth angles is expected to have the same size as the
                list of polar angles; otherwise, a ValueError is raised.
                Also, the two lists of angles have to be coherent in
                order. That is, it is expected that the order in the list
                of azimuth angles corresponds to a distinct sequence of
                neighbors. And, this sequence has to equal the sequence
                of neighbors in the list of polar angles.
        """
    if len(thetas) != len(phis):
        raise ValueError('List of polar and azimuthal angles have to be equal!')
    self._pow_sin_t.clear()
    self._pow_cos_t.clear()
    self._sin_n_p.clear()
    self._cos_n_p.clear()
    self._pow_sin_t[1] = [sin(float(t)) for t in thetas]
    self._pow_cos_t[1] = [cos(float(t)) for t in thetas]
    self._sin_n_p[1] = [sin(float(p)) for p in phis]
    self._cos_n_p[1] = [cos(float(p)) for p in phis]
    for idx in range(2, self._max_trig_order + 1):
        self._pow_sin_t[idx] = [e[0] * e[1] for e in zip(self._pow_sin_t[idx - 1], self._pow_sin_t[1])]
        self._pow_cos_t[idx] = [e[0] * e[1] for e in zip(self._pow_cos_t[idx - 1], self._pow_cos_t[1])]
        self._sin_n_p[idx] = [sin(float(idx) * float(p)) for p in phis]
        self._cos_n_p[idx] = [cos(float(idx) * float(p)) for p in phis]