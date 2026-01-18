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
def get_nn_data(self, structure: Structure, n: int, length=None):
    """
        The main logic of the method to compute near neighbor.

        Args:
            structure: (Structure) enclosing structure object
            n: (int) index of target site to get NN info for
            length: (int) if set, will return a fixed range of CN numbers

        Returns:
            a namedtuple (NNData) object that contains:
            - all near neighbor sites with weights
            - a dict of CN -> weight
            - a dict of CN -> associated near neighbor sites
        """
    length = length or self.fingerprint_length
    target = None
    if self.cation_anion:
        target = []
        m_oxi = structure[n].specie.oxi_state
        for site in structure:
            oxi_state = getattr(site.specie, 'oxi_state', None)
            if oxi_state is not None and oxi_state * m_oxi <= 0:
                target.append(site.specie)
        if not target:
            raise ValueError('No valid targets for site within cation_anion constraint!')
    cutoff = self.search_cutoff
    vnn = VoronoiNN(weight='solid_angle', targets=target, cutoff=cutoff)
    nn = vnn.get_nn_info(structure, n)
    if self.porous_adjustment:
        for x in nn:
            x['weight'] *= x['poly_info']['solid_angle'] / x['poly_info']['area']
    if self.x_diff_weight > 0:
        for entry in nn:
            X1 = structure[n].specie.X
            X2 = entry['site'].specie.X
            if math.isnan(X1) or math.isnan(X2):
                chemical_weight = 1
            else:
                chemical_weight = 1 + self.x_diff_weight * math.sqrt(abs(X1 - X2) / 3.3)
            entry['weight'] = entry['weight'] * chemical_weight
    nn = sorted(nn, key=lambda x: x['weight'], reverse=True)
    if nn[0]['weight'] == 0:
        return self.transform_to_length(self.NNData([], {0: 1.0}, {0: []}), length)
    highest_weight = nn[0]['weight']
    for entry in nn:
        entry['weight'] = entry['weight'] / highest_weight
    if self.distance_cutoffs:
        r1 = _get_radius(structure[n])
        for entry in nn:
            r2 = _get_radius(entry['site'])
            if r1 > 0 and r2 > 0:
                diameter = r1 + r2
            else:
                warnings.warn('CrystalNN: cannot locate an appropriate radius, covalent or atomic radii will be used, this can lead to non-optimal results.')
                diameter = _get_default_radius(structure[n]) + _get_default_radius(entry['site'])
            dist = np.linalg.norm(structure[n].coords - entry['site'].coords)
            dist_weight: float = 0
            cutoff_low = diameter + self.distance_cutoffs[0]
            cutoff_high = diameter + self.distance_cutoffs[1]
            if dist <= cutoff_low:
                dist_weight = 1
            elif dist < cutoff_high:
                dist_weight = (math.cos((dist - cutoff_low) / (cutoff_high - cutoff_low) * math.pi) + 1) * 0.5
            entry['weight'] = entry['weight'] * dist_weight
    nn = sorted(nn, key=lambda x: x['weight'], reverse=True)
    if nn[0]['weight'] == 0:
        return self.transform_to_length(self.NNData([], {0: 1.0}, {0: []}), length)
    for entry in nn:
        entry['weight'] = round(entry['weight'], 3)
        del entry['poly_info']
    nn = [x for x in nn if x['weight'] > 0]
    dist_bins: list[float] = []
    for entry in nn:
        if not dist_bins or dist_bins[-1] != entry['weight']:
            dist_bins.append(entry['weight'])
    dist_bins.append(0)
    cn_weights = {}
    cn_nninfo = {}
    for idx, val in enumerate(dist_bins):
        if val != 0:
            nn_info = []
            for entry in nn:
                if entry['weight'] >= val:
                    nn_info.append(entry)
            cn = len(nn_info)
            cn_nninfo[cn] = nn_info
            cn_weights[cn] = self._semicircle_integral(dist_bins, idx)
    cn0_weight = 1 - sum(cn_weights.values())
    if cn0_weight > 0:
        cn_nninfo[0] = []
        cn_weights[0] = cn0_weight
    return self.transform_to_length(self.NNData(nn, cn_weights, cn_nninfo), length)