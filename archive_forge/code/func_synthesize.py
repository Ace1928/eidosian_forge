import numpy as np
from pygsp import utils
from . import approximations
def synthesize(self, s, method='chebyshev', order=30):
    """Convenience wrapper around :meth:`filter`.

        Will be an alias to `adjoint().filter()` in the future.
        """
    if s.shape[-1] != self.Nf:
        raise ValueError('Last dimension (#features) should be the number of filters Nf = {}, got {}.'.format(self.Nf, s.shape))
    return self.filter(s, method, order)