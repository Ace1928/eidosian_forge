import numpy as np  # type: ignore
from typing import Tuple, Optional
def rotmat(p, q):
    """Return a (left multiplying) matrix that rotates p onto q.

    :param p: moving vector
    :type p: L{Vector}

    :param q: fixed vector
    :type q: L{Vector}

    :return: rotation matrix that rotates p onto q
    :rtype: 3x3 NumPy array

    Examples
    --------
    >>> from Bio.PDB.vectors import rotmat
    >>> p, q = Vector(1, 2, 3), Vector(2, 3, 5)
    >>> r = rotmat(p, q)
    >>> print(q)
    <Vector 2.00, 3.00, 5.00>
    >>> print(p)
    <Vector 1.00, 2.00, 3.00>
    >>> p.left_multiply(r)
    <Vector 1.21, 1.82, 3.03>

    """
    rot = np.dot(refmat(q, -p), refmat(p, -p))
    return rot