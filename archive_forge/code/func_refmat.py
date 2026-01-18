import numpy as np  # type: ignore
from typing import Tuple, Optional
def refmat(p, q):
    """Return a (left multiplying) matrix that mirrors p onto q.

    :type p,q: L{Vector}
    :return: The mirror operation, a 3x3 NumPy array.

    Examples
    --------
    >>> from Bio.PDB.vectors import refmat
    >>> p, q = Vector(1, 2, 3), Vector(2, 3, 5)
    >>> mirror = refmat(p, q)
    >>> qq = p.left_multiply(mirror)
    >>> print(q)
    <Vector 2.00, 3.00, 5.00>
    >>> print(qq)
    <Vector 1.21, 1.82, 3.03>

    """
    p = p.normalized()
    q = q.normalized()
    if (p - q).norm() < 1e-05:
        return np.identity(3)
    pq = p - q
    pq.normalize()
    b = pq.get_array()
    b.shape = (3, 1)
    i = np.identity(3)
    ref = i - 2 * np.dot(b, np.transpose(b))
    return ref