import math
def length2(self):
    """Returns the length of a vector squared.

        >>> Vector(10, 10).length2()
        200
        >>> pos = (10, 10)
        >>> Vector(pos).length2()
        200

        """
    return self[0] ** 2 + self[1] ** 2