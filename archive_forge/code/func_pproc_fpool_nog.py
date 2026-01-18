import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
def pproc_fpool_nog(self):
    """
        Process all field functions with no constraints supplied in parallel.
        """
    self.wfield.func
    fpool_l = []
    for v in self.fpool:
        fpool_l.append(v.x_a)
    F = self._mapwrapper(self.wfield.func, fpool_l)
    for va, f in zip(fpool_l, F):
        vt = tuple(va)
        self[vt].f = f
        self.nfev += 1
    self.fpool = set()