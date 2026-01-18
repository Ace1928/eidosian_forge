import math
@staticmethod
def in_bbox(point, a, b):
    """Return True if `point` is in the bounding box defined by `a`
        and `b`.

        >>> bmin = (0, 0)
        >>> bmax = (100, 100)
        >>> Vector.in_bbox((50, 50), bmin, bmax)
        True
        >>> Vector.in_bbox((647, -10), bmin, bmax)
        False

        """
    return (point[0] <= a[0] and point[0] >= b[0] or (point[0] <= b[0] and point[0] >= a[0])) and (point[1] <= a[1] and point[1] >= b[1] or (point[1] <= b[1] and point[1] >= a[1]))