import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
def pproc_gpool(self):
    """Process all constraints in parallel."""
    gpool_l = []
    for v in self.gpool:
        gpool_l.append(v.x_a)
    G = self._mapwrapper(self.wgcons.gcons, gpool_l)
    for v, g in zip(self.gpool, G):
        v.feasible = g