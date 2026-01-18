from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def can_straighten_bend(arc, bend, check_embedded=True, bend_matrix=None):
    """
    Given a "bend" of three points (a, b, c) and an arc (u, v)
    determine if we can isotope the PL arc (a, b, c) to (a, c) along
    the triangle (a, b, c) without passing through (u, v).

    >>> a = Vector3([-1, 0, 1])
    >>> b = Vector3([ 0, 0, 1])
    >>> c = Vector3([ 0, 1, 0])
    >>> o = Vector3([ 0, 0, 0])
    >>> u = Vector3([-1, 2, 1])
    >>> can_straighten_bend((o, -a), (a, b, c))
    True
    >>> can_straighten_bend((o, c), (a, 2*a, 3*a))
    True
    >>> can_straighten_bend((o, c), (a, b, c))
    True
    >>> can_straighten_bend((a, (a + b + c)/3), (a, b, c))
    False
    >>> can_straighten_bend((c, (a + c)/2), (a, b, c))
    False
    >>> can_straighten_bend((c, a), (a, b, c))
    False
    >>> can_straighten_bend((o, u), (a, b, c))
    False
    >>> can_straighten_bend((a + b, a + c), (a, b, c))
    True
    >>> can_straighten_bend((o, a + c), (a, b, c))
    False
    >>> can_straighten_bend((a, c - b), (a, b, c))
    True
    >>> M = standardize_bend_matrix(a, b, c); M.det()
    1/2
    >>> can_straighten_bend((o, -a), (a, b, c), bend_matrix=M)
    True
    """
    u, v = arc
    a, b, c = bend
    if hasattr(u, 'to_3d_point'):
        u, v = (u.to_3d_point(), v.to_3d_point())
        a, b, c = (a.to_3d_point(), b.to_3d_point(), c.to_3d_point())
    if u == v or a == b or a == c or (b == c):
        raise ValueError('Input includes 0 length segments')
    if check_embedded:
        if segments_meet_not_at_endpoint((a, b), (b, c)) or segments_meet_not_at_endpoint((a, b), (u, v)) or segments_meet_not_at_endpoint((b, c), (u, v)) or point_meets_interior_of_segment(b, (u, v)) or (u == b) or (v == b):
            raise ValueError('Input not embedded')
    for i in range(3):
        arc_cor = (u[i], v[i])
        bend_cor = (a[i], b[i], c[i])
        if max(arc_cor) < min(bend_cor) or min(arc_cor) > max(bend_cor):
            return True
    if colinear(a, b, c):
        return True
    if bend_matrix is None:
        bend_matrix = standardize_bend_matrix(a, b, c)
    a, c = (Vector3([1, 0, 0]), Vector3([0, 1, 0]))
    u, v = (bend_matrix * (u - b), bend_matrix * (v - b))
    if u[2] * v[2] > 0:
        return True
    if u[2] == v[2] == 0:
        if u == a and v == c or (u == c and v == a):
            return False
        u_in_tri = u[0] > 0 and u[1] > 0 and (u[0] + u[1] <= 1)
        v_in_tri = v[0] > 0 and v[1] > 0 and (v[0] + v[1] <= 1)
        return not (u_in_tri or v_in_tri)
    t = u[2] / (u[2] - v[2])
    p = (1 - t) * u + t * v
    assert 0 <= t <= 1 and p[2] == 0
    return not (p[0] > 0 and p[1] > 0 and (p[0] + p[1] <= 1))