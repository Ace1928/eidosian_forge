import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
def maximiser(self):
    """
        Check whether this vertex is strictly greater than all its
        neighbours.
        """
    if self.check_max:
        self._max = all((self.f > v.f for v in self.nn))
        self.check_max = False
    return self._max