import numpy as np
from itertools import combinations_with_replacement
from math import erf
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc
def take_individual_rdf(index, unique_type):
    rdf = np.zeros(nbins)
    if self.dimensions == 3:
        weights = 1.0 / surface_area_3d(bindist)
    elif self.dimensions == 2:
        weights = 1.0 / surface_area_2d(bindist, pos[index])
    elif self.dimensions == 1:
        weights = 1.0 / surface_area_1d(bindist, pos[index])
    elif self.dimensions == 0:
        weights = 1.0 / surface_area_0d(bindist)
    weights /= self.binwidth
    indices, offsets = nl.get_neighbors(index)
    valid = np.where(num[indices] == unique_type)
    p = pos[indices[valid]] + np.dot(offsets[valid], cell)
    r = cdist(p, [pos[index]])
    bins = np.floor(r / self.binwidth)
    for i in range(-m, m + 1):
        newbins = bins + i
        valid = np.where((newbins >= 0) & (newbins < nbins))
        valid_bins = newbins[valid].astype(int)
        values = weights[valid_bins]
        c = 0.25 * np.sqrt(2) * self.binwidth * 1.0 / self.sigma
        values *= 0.5 * erf(c * (2 * i + 1)) - 0.5 * erf(c * (2 * i - 1))
        values /= smearing_norm
        for j, valid_bin in enumerate(valid_bins):
            rdf[valid_bin] += values[j]
    rdf /= len(typedic[unique_type]) * 1.0 / volume
    return rdf