import numpy as np
from pygsp import utils
def gft_windowed_normalized(self, g, f, lowmemory=True):
    """Normalized windowed graph Fourier transform.

        Parameters
        ----------
        g : ndarray
            Window.
        f : ndarray
            Graph signal in the vertex domain.
        lowmemory : bool
            Use less memory. (default = True)

        Returns
        -------
        C : ndarray
            Coefficients.

        """
    raise NotImplementedError('Current implementation is not working.')
    N = self.N
    U = self.U
    if lowmemory:
        Frame = self._frame_matrix(g, normalize=True)
        C = np.dot(Frame.T, f)
        C = np.reshape(C, (N, N), order='F')
    else:
        ghat = np.dot(U.T, g)
        Ftrans = np.sqrt(N) * np.dot(U, np.kron(np.ones((1, N)), ghat) * U.T)
        C = np.empty((N, N))
        for i in range(N):
            atoms = np.kron(np.ones(N), 1.0 / U[:, 0]) * U * np.kron(np.ones(N), Ftrans[:, i]).T
            atoms /= np.kron(np.ones(N), np.sqrt(np.sum(np.abs(atoms), axis=0)))
            C[:, i] = np.dot(atoms, f)
    return C