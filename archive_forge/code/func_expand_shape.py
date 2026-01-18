import numpy as np
from .base import product
from .. import h5s, h5r, _selector
def expand_shape(self, source_shape):
    if not source_shape == self.array_shape:
        raise TypeError('Broadcasting is not supported for complex selections')
    return source_shape