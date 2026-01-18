import cupy
def poly(A):
    """np.poly replacement for 2D A. Otherwise, use cupy.poly."""
    sh = A.shape
    if not (len(sh) == 2 and sh[0] == sh[1] and (sh[0] != 0)):
        raise ValueError('input must be a non-empty square 2d array.')
    import numpy as np
    seq_of_zeros = np.linalg.eigvals(A.get())
    dt = seq_of_zeros.dtype
    a = np.ones((1,), dtype=dt)
    for zero in seq_of_zeros:
        a = np.convolve(a, np.r_[1, -zero], mode='full')
    if issubclass(a.dtype.type, cupy.complexfloating):
        roots = np.asarray(seq_of_zeros, dtype=complex)
        if np.all(np.sort(roots) == np.sort(roots.conjugate())):
            a = a.real.copy()
    return cupy.asarray(a)