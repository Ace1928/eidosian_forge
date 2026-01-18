import numpy as np
from pygsp import utils
def gft_windowed(self, g, f, lowmemory=True):
    """Windowed graph Fourier transform.

        Parameters
        ----------
        g : ndarray or Filter
            Window (graph signal or kernel).
        f : ndarray
            Graph signal in the vertex domain.
        lowmemory : bool
            Use less memory (default=True).

        Returns
        -------
        C : ndarray
            Coefficients.

        """
    raise NotImplementedError('Current implementation is not working.')
    N = self.N
    Nf = np.shape(f)[1]
    U = self.U
    if isinstance(g, list):
        g = self.igft(g[0](self.e))
    elif hasattr(g, '__call__'):
        g = self.igft(g(self.e))
    if not lowmemory:
        Frame = self._frame_matrix(g, normalize=False)
        C = np.dot(Frame.T, f)
        C = np.reshape(C, (N, N, Nf), order='F')
    else:
        ghat = np.dot(U.T, g)
        Ftrans = np.sqrt(N) * np.dot(U, np.kron(np.ones(N), ghat) * U.T)
        C = np.empty((N, N))
        for j in range(Nf):
            for i in range(N):
                C[:, i, j] = (np.kron(np.ones(N), 1.0 / U[:, 0]) * U * np.dot(np.kron(np.ones(N), Ftrans[:, i])).T, f[:, j])
    return C