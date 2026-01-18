import numpy as np  # type: ignore
from typing import Tuple, Optional
def rotaxis2m(theta, vector):
    """Calculate left multiplying rotation matrix.

    Calculate a left multiplying rotation matrix that rotates
    theta rad around vector.

    :type theta: float
    :param theta: the rotation angle

    :type vector: L{Vector}
    :param vector: the rotation axis

    :return: The rotation matrix, a 3x3 NumPy array.

    Examples
    --------
    >>> from numpy import pi
    >>> from Bio.PDB.vectors import rotaxis2m
    >>> from Bio.PDB.vectors import Vector
    >>> m = rotaxis2m(pi, Vector(1, 0, 0))
    >>> Vector(1, 2, 3).left_multiply(m)
    <Vector 1.00, -2.00, -3.00>

    """
    vector = vector.normalized()
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    x, y, z = vector.get_array()
    rot = np.zeros((3, 3))
    rot[0, 0] = t * x * x + c
    rot[0, 1] = t * x * y - s * z
    rot[0, 2] = t * x * z + s * y
    rot[1, 0] = t * x * y + s * z
    rot[1, 1] = t * y * y + c
    rot[1, 2] = t * y * z - s * x
    rot[2, 0] = t * x * z - s * y
    rot[2, 1] = t * y * z + s * x
    rot[2, 2] = t * z * z + c
    return rot