import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
def proc_gpool(self):
    """Process all constraints."""
    if self.g_cons is not None:
        for v in self.gpool:
            self.feasibility_check(v)
    self.gpool = set()