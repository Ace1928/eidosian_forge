import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
def minimiser(self):
    """Check whether this vertex is strictly less than all its
           neighbours"""
    if self.check_min:
        self._min = all((self.f < v.f for v in self.nn))
        self.check_min = False
    return self._min