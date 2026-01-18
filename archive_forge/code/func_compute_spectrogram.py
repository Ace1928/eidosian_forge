import numpy as np
from pygsp import filters, utils
def compute_spectrogram(G, atom=None, M=100, **kwargs):
    """
    Compute the norm of the Tig for all nodes with a kernel shifted along the
    spectral axis.

    Parameters
    ----------
    G : Graph
        Graph on which to compute the spectrogram.
    atom : func
        Kernel to use in the spectrogram (default = exp(-M*(x/lmax)²)).
    M : int (optional)
        Number of samples on the spectral scale. (default = 100)
    kwargs: dict
        Additional parameters to be passed to the
        :func:`pygsp.filters.Filter.filter` method.
    """
    if not atom:

        def atom(x):
            return np.exp(-M * (x / G.lmax) ** 2)
    scale = np.linspace(0, G.lmax, M)
    spectr = np.empty((G.N, M))
    for shift_idx in range(M):
        shift_filter = filters.Filter(G, lambda x: atom(x - scale[shift_idx]))
        tig = compute_norm_tig(shift_filter, **kwargs).squeeze() ** 2
        spectr[:, shift_idx] = tig
    G.spectr = spectr
    return spectr