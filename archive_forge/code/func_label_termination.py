from __future__ import annotations
import logging
import warnings
from fractions import Fraction
from functools import reduce
from itertools import chain, combinations, product
from math import cos, floor, gcd
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.fractions import lcm
from numpy.testing import assert_allclose
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def label_termination(slab: Structure) -> str:
    """Labels the slab surface termination."""
    frac_coords = slab.frac_coords
    n = len(frac_coords)
    if n == 1:
        form = slab.reduced_formula
        sp_symbol = SpacegroupAnalyzer(slab, symprec=0.1).get_space_group_symbol()
        return f'{form}_{sp_symbol}_{len(slab)}'
    dist_matrix = np.zeros((n, n))
    h = slab.lattice.c
    for ii, jj in combinations(list(range(n)), 2):
        if ii != jj:
            cdist = frac_coords[ii][2] - frac_coords[jj][2]
            cdist = abs(cdist - round(cdist)) * h
            dist_matrix[ii, jj] = cdist
            dist_matrix[jj, ii] = cdist
    condensed_m = squareform(dist_matrix)
    z = linkage(condensed_m)
    clusters = fcluster(z, 0.25, criterion='distance')
    clustered_sites: dict[int, list[Site]] = {c: [] for c in clusters}
    for idx, cluster in enumerate(clusters):
        clustered_sites[cluster].append(slab[idx])
    plane_heights = {np.average(np.mod([s.frac_coords[2] for s in sites], 1)): c for c, sites in clustered_sites.items()}
    top_plane_cluster = sorted(plane_heights.items(), key=lambda x: x[0])[-1][1]
    top_plane_sites = clustered_sites[top_plane_cluster]
    top_plane = Structure.from_sites(top_plane_sites)
    sp_symbol = SpacegroupAnalyzer(top_plane, symprec=0.1).get_space_group_symbol()
    form = top_plane.reduced_formula
    return f'{form}_{sp_symbol}_{len(top_plane)}'