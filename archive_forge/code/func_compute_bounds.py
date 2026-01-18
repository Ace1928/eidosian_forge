import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def compute_bounds(self, scs):
    spec = self._slicespec2boundsspec(self, scs)
    return BoundingBox(points=spec)