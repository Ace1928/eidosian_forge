import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
def feasibility_check(self, v):
    v.feasible = True
    for g, args in zip(self.g_cons, self.g_cons_args):
        if np.any(g(v.x_a, *args) < 0.0):
            v.f = np.inf
            v.feasible = False
            break