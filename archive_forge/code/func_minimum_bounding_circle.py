from kivy.vector import Vector
def minimum_bounding_circle(points):
    """
    Returns the minimum bounding circle for a set of points.

    For a description of the problem being solved, see the `Smallest Circle
    Problem <http://en.wikipedia.org/wiki/Smallest_circle_problem>`_.

    The function uses Applet's Algorithm, the runtime is ``O(h^3, *n)``,
    where h is the number of points in the convex hull of the set of points.
    **But** it runs in linear time in almost all real world cases.
    See: http://tinyurl.com/6e4n5yb

    :Parameters:
        `points`: iterable
            A list of points (2 tuple with x,y coordinates)

    :Return:
        A tuple that defines the circle:
            * The first element in the returned tuple is the center (x, y)
            * The second the radius (float)

    """
    points = [Vector(p[0], p[1]) for p in points]
    if len(points) == 1:
        return ((points[0].x, points[0].y), 0.0)
    if len(points) == 2:
        p1, p2 = points
        return ((p1 + p2) * 0.5, ((p1 - p2) * 0.5).length())
    P = min(points, key=lambda p: p.y)

    def x_axis_angle(q):
        if q == P:
            return 10000000000.0
        return abs((q - P).angle((1, 0)))
    Q = min(points, key=x_axis_angle)
    for p in points:

        def angle_pq(r):
            if r in (P, Q):
                return 10000000000.0
            return abs((r - P).angle(r - Q))
        R = min(points, key=angle_pq)
        if angle_pq(R) > 90.0:
            return ((P + Q) * 0.5, ((P - Q) * 0.5).length())
        if abs((R - P).angle(Q - P)) > 90:
            P = R
            continue
        if abs((P - Q).angle(R - Q)) > 90:
            Q = R
            continue
        break
    return circumcircle(P, Q, R)