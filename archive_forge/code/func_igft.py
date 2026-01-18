import numpy as np
from pygsp import utils
def igft(self, s_hat):
    """Compute the inverse graph Fourier transform.

        The inverse graph Fourier transform of a Fourier domain signal
        :math:`\\hat{s}` is defined as

        .. math:: s = U \\hat{s},

        where :math:`U` is the Fourier basis :attr:`U`.

        Parameters
        ----------
        s_hat : ndarray
            Graph signal in the Fourier domain.

        Returns
        -------
        s : ndarray
            Representation of s_hat in the vertex domain.

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.compute_fourier_basis()
        >>> s_hat = np.random.normal(size=(G.N, 5, 1))
        >>> s = G.igft(s_hat)
        >>> s_hat_star = G.gft(s)
        >>> np.all((s_hat - s_hat_star) < 1e-10)
        True

        """
    if s_hat.shape[0] != self.N:
        raise ValueError('First dimension should be the number of nodes G.N = {}, got {}.'.format(self.N, s_hat.shape))
    return np.tensordot(self.U, s_hat, ([1], [0]))