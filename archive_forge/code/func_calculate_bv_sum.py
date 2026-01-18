from __future__ import annotations
import collections
import functools
import operator
import os
from math import exp, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.serialization import loadfn
from pymatgen.core import Element, Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def calculate_bv_sum(site, nn_list, scale_factor=1.0):
    """Calculates the BV sum of a site.

    Args:
        site (PeriodicSite): The central site to calculate the bond valence
        nn_list ([Neighbor]): A list of namedtuple Neighbors having "distance"
            and "site" attributes
        scale_factor (float): A scale factor to be applied. This is useful for
            scaling distance, esp in the case of calculation-relaxed structures
            which may tend to under (GGA) or over bind (LDA).
    """
    el1 = Element(site.specie.symbol)
    bv_sum = 0
    for nn in nn_list:
        el2 = Element(nn.specie.symbol)
        if (el1 in ELECTRONEG or el2 in ELECTRONEG) and el1 != el2:
            r1 = BV_PARAMS[el1]['r']
            r2 = BV_PARAMS[el2]['r']
            c1 = BV_PARAMS[el1]['c']
            c2 = BV_PARAMS[el2]['c']
            R = r1 + r2 - r1 * r2 * (sqrt(c1) - sqrt(c2)) ** 2 / (c1 * r1 + c2 * r2)
            vij = exp((R - nn.nn_distance * scale_factor) / 0.31)
            bv_sum += vij * (1 if el1.X < el2.X else -1)
    return bv_sum