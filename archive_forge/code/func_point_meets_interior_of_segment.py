from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def point_meets_interior_of_segment(u, ab):
    """
    >>> p = Vector3([1, 1, 0])
    >>> q = Vector3([3, -1,0])
    >>> a = Vector3([2, 0, 0])
    >>> b = Vector3([0, 2, 0])
    >>> c = Vector3([1, 0, 0])
    >>> o = Vector3([0, 0, 0])
    >>> point_meets_interior_of_segment(p, (a, b))
    True
    >>> point_meets_interior_of_segment(q, (a, b))
    False
    >>> point_meets_interior_of_segment(p, (b, c))
    False
    >>> point_meets_interior_of_segment(2*a/3, (a, c))
    True
    >>> point_meets_interior_of_segment(o, (a, b))
    False
    """
    a, b = ab
    if a == b:
        raise ValueError('Segment is degenerate')
    if not are_parallel(b - a, u - a):
        return False
    t = find_convex_combination(a, b, u)
    return 0 < t < 1