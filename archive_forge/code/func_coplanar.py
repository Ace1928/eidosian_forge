from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def coplanar(a, b, c, d):
    """
    >>> vecs = Matrix([(1, 0, 1), (-1, 0, 2), (0, 1, 0), (1, 3, 0)])
    >>> coplanar(*vecs.rows())
    False
    >>> coplanar(vecs[0], vecs[1], vecs[2], vecs[0])
    True
    """
    return Matrix([b - a, c - a, d - a]).det() == 0