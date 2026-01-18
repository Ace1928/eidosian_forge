import math
import cupy
from cupy.linalg import _util
def expm(a):
    """Compute the matrix exponential.

    Parameters
    ----------
    a : ndarray, 2D

    Returns
    -------
    matrix exponential of `a`

    Notes
    -----
    Uses (a simplified) version of Algorithm 2.3 of [1]_:
    a [13 / 13] Pade approximant with scaling and squaring.

    Simplifications:

        * we always use a [13/13] approximate
        * no matrix balancing

    References
    ----------
    .. [1] N. Higham, SIAM J. MATRIX ANAL. APPL. Vol. 26(4), p. 1179 (2005)
       https://doi.org/10.1137/04061101X

    """
    if a.size == 0:
        return cupy.zeros((0, 0), dtype=a.dtype)
    n = a.shape[0]
    mu = cupy.diag(a).sum() / n
    A = a - cupy.eye(n) * mu
    nrmA = cupy.linalg.norm(A, ord=1).item()
    scale = nrmA > th13
    if scale:
        s = int(math.ceil(math.log2(float(nrmA) / th13))) + 1
    else:
        s = 1
    A /= 2 ** s
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4
    E = cupy.eye(A.shape[0])
    u1, u2, v1, v2 = _expm_inner(E, A, A2, A4, A6, cupy.asarray(b))
    u = A @ (A6 @ u1 + u2)
    v = A6 @ v1 + v2
    r13 = cupy.linalg.solve(-u + v, u + v)
    x = r13
    for _ in range(s):
        x = x @ x
    x *= math.exp(mu)
    return x