import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def circular_sum(self, other, n=0):
    """
        Glue two tangles together to form a link by gluing them vertically and then taking
        the braid closure (the ``Tangle.denominator_closure()``).
        The second tangle is rotated clockwise by n strands using ``Tangle.circular_rotate()``.
        """
    Am, An = self.boundary
    Bm, Bn = self.boundary
    if (Am, An) != (Bn, Bm):
        raise ValueError('Tangles must have compatible boundary shapes')
    return (self * other.circular_rotate(n)).denominator_closure()