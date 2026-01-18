import numpy as np
from itertools import combinations_with_replacement
from math import erf
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc
def get_local_orders(self, a):
    """ Returns the local orders of all the atoms."""
    a_top = a[-self.n_top:]
    key = 'individual_fingerprints'
    if key in a.info and (not self.recalculate):
        fp, typedic = self._json_decode(*a.info[key])
    else:
        fp, typedic = self._take_fingerprints(a_top, individual=True)
        a.info[key] = self._json_encode(fp, typedic)
    volume, pmin, pmax, qmin, qmax = self._get_volume(a_top)
    return self._calculate_local_orders(fp, typedic, volume)