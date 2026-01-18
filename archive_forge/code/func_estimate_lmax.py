from collections import Counter
import numpy as np
from scipy import sparse
from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5
def estimate_lmax(self, recompute=False):
    """Estimate the Laplacian's largest eigenvalue (cached).

        The result is cached and accessible by the :attr:`lmax` property.

        Exact value given by the eigendecomposition of the Laplacian, see
        :func:`compute_fourier_basis`. That estimation is much faster than the
        eigendecomposition.

        Parameters
        ----------
        recompute : boolean
            Force to recompute the largest eigenvalue. Default is false.

        Notes
        -----
        Runs the implicitly restarted Lanczos method with a large tolerance,
        then increases the calculated largest eigenvalue by 1 percent. For much
        of the PyGSP machinery, we need to approximate wavelet kernels on an
        interval that contains the spectrum of L. The only cost of using a
        larger interval is that the polynomial approximation over the larger
        interval may be a slightly worse approximation on the actual spectrum.
        As this is a very mild effect, it is not necessary to obtain very tight
        bounds on the spectrum of L.

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.compute_fourier_basis()
        >>> print('{:.2f}'.format(G.lmax))
        13.78
        >>> G = graphs.Logo()
        >>> G.estimate_lmax(recompute=True)
        >>> print('{:.2f}'.format(G.lmax))
        13.92

        """
    if hasattr(self, '_lmax') and (not recompute):
        return
    try:
        lmax = sparse.linalg.eigsh(self.L, k=1, tol=0.005, ncv=min(self.N, 10), return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1.01
    except sparse.linalg.ArpackNoConvergence:
        self.logger.warning('Lanczos method did not converge. Using an alternative method.')
        if self.lap_type == 'normalized':
            lmax = 2
        elif self.lap_type == 'combinatorial':
            lmax = 2 * np.max(self.dw)
        else:
            raise ValueError('Unknown Laplacian type {}'.format(self.lap_type))
    self._lmax = lmax