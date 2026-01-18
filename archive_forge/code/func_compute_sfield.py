import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
def compute_sfield(self, v):
    """Compute the scalar field values of a vertex object `v`.

        Parameters
        ----------
        v : VertexBase or VertexScalarField object
        """
    try:
        v.f = self.field(v.x_a, *self.field_args)
        self.nfev += 1
    except AttributeError:
        v.f = np.inf
    if np.isnan(v.f):
        v.f = np.inf