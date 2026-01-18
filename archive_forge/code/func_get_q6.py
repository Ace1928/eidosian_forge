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
def get_q6(self, thetas=None, phis=None):
    """
        Calculates the value of the bond orientational order parameter of
        weight l=6. If the function is called with non-empty lists of
        polar and azimuthal angles the corresponding trigonometric terms
        are computed afresh. Otherwise, it is expected that the
        compute_trigonometric_terms function has been just called.

        Args:
            thetas ([float]): polar angles of all neighbors in radians.
            phis ([float]): azimuth angles of all neighbors in radians.

        Returns:
            float: bond orientational order parameter of weight l=6
                corresponding to the input angles thetas and phis.
        """
    if thetas is not None and phis is not None:
        self.compute_trigonometric_terms(thetas, phis)
    n_nn = len(self._pow_sin_t[1])
    n_nn_range = range(n_nn)
    i64 = 1 / 64.0
    i32 = 1 / 32.0
    i32_3 = 3 / 32.0
    i16 = 1 / 16.0
    sqrt_3003_pi = sqrt(3003 / pi)
    sqrt_1001_pi = sqrt(1001 / pi)
    sqrt_91_2pi = sqrt(91 / (2 * pi))
    sqrt_1365_pi = sqrt(1365 / pi)
    sqrt_273_2pi = sqrt(273 / (2 * pi))
    sqrt_13_pi = sqrt(13 / pi)
    pre_y_6_6 = [i64 * sqrt_3003_pi * val for val in self._pow_sin_t[6]]
    pre_y_6_5 = [i32_3 * sqrt_1001_pi * val[0] * val[1] for val in zip(self._pow_sin_t[5], self._pow_cos_t[1])]
    pre_y_6_4 = [i32_3 * sqrt_91_2pi * val[0] * (11 * val[1] - 1.0) for val in zip(self._pow_sin_t[4], self._pow_cos_t[2])]
    pre_y_6_3 = [i32 * sqrt_1365_pi * val[0] * (11 * val[1] - 3 * val[2]) for val in zip(self._pow_sin_t[3], self._pow_cos_t[3], self._pow_cos_t[1])]
    pre_y_6_2 = [i64 * sqrt_1365_pi * val[0] * (33 * val[1] - 18 * val[2] + 1.0) for val in zip(self._pow_sin_t[2], self._pow_cos_t[4], self._pow_cos_t[2])]
    pre_y_6_1 = [i16 * sqrt_273_2pi * val[0] * (33 * val[1] - 30 * val[2] + 5 * val[3]) for val in zip(self._pow_sin_t[1], self._pow_cos_t[5], self._pow_cos_t[3], self._pow_cos_t[1])]
    acc = 0.0
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real += pre_y_6_6[idx] * self._cos_n_p[6][idx]
        imag -= pre_y_6_6[idx] * self._sin_n_p[6][idx]
    acc += real * real + imag * imag
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real += pre_y_6_5[idx] * self._cos_n_p[5][idx]
        imag -= pre_y_6_5[idx] * self._sin_n_p[5][idx]
    acc += real * real + imag * imag
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real += pre_y_6_4[idx] * self._cos_n_p[4][idx]
        imag -= pre_y_6_4[idx] * self._sin_n_p[4][idx]
    acc += real * real + imag * imag
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real += pre_y_6_3[idx] * self._cos_n_p[3][idx]
        imag -= pre_y_6_3[idx] * self._sin_n_p[3][idx]
    acc += real * real + imag * imag
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real += pre_y_6_2[idx] * self._cos_n_p[2][idx]
        imag -= pre_y_6_2[idx] * self._sin_n_p[2][idx]
    acc += real * real + imag * imag
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real += pre_y_6_1[idx] * self._cos_n_p[1][idx]
        imag -= pre_y_6_1[idx] * self._sin_n_p[1][idx]
    acc += real * real + imag * imag
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real += i32 * sqrt_13_pi * (231 * self._pow_cos_t[6][idx] - 315 * self._pow_cos_t[4][idx] + 105 * self._pow_cos_t[2][idx] - 5.0)
    acc += real * real
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real -= pre_y_6_1[idx] * self._cos_n_p[1][idx]
        imag -= pre_y_6_1[idx] * self._sin_n_p[1][idx]
    acc += real * real + imag * imag
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real += pre_y_6_2[idx] * self._cos_n_p[2][idx]
        imag += pre_y_6_2[idx] * self._sin_n_p[2][idx]
    acc += real * real + imag * imag
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real -= pre_y_6_3[idx] * self._cos_n_p[3][idx]
        imag -= pre_y_6_3[idx] * self._sin_n_p[3][idx]
    acc += real * real + imag * imag
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real += pre_y_6_4[idx] * self._cos_n_p[4][idx]
        imag += pre_y_6_4[idx] * self._sin_n_p[4][idx]
    acc += real * real + imag * imag
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real -= pre_y_6_5[idx] * self._cos_n_p[5][idx]
        imag -= pre_y_6_5[idx] * self._sin_n_p[5][idx]
    acc += real * real + imag * imag
    real = 0.0
    imag = 0.0
    for idx in n_nn_range:
        real += pre_y_6_6[idx] * self._cos_n_p[6][idx]
        imag += pre_y_6_6[idx] * self._sin_n_p[6][idx]
    acc += real * real + imag * imag
    return sqrt(4 * pi * acc / (13 * float(n_nn * n_nn)))