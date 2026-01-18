import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
def proc_fpool_nog(self):
    """Process all field functions with no constraints supplied."""
    for v in self.fpool:
        self.compute_sfield(v)
    self.fpool = set()