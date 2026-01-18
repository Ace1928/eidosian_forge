import numpy as np
from pygsp import utils
@property
def mu(self):
    """Coherence of the Fourier basis.

        Is computed by :func:`compute_fourier_basis`.
        """
    return self._check_fourier_properties('mu', 'Fourier basis coherence')