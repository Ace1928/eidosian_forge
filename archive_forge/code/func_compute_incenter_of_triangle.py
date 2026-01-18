from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def compute_incenter_of_triangle(idealPoints):
    """
    Computes incenter of the triangle spanned by three ideal points::

        sage: from sage.all import CIF
        sage: z0 = Infinity
        sage: z1 = CIF(0)
        sage: z2 = CIF(1)
        sage: compute_incenter_of_triangle([z0, z1, z2]) # doctest: +NUMERIC12
        FinitePoint(0.50000000000000000?, 0.866025403784439?)
    """
    if not len(idealPoints) == 3:
        raise Exception('Expected 3 ideal points.')
    transformedIdealPoints, inv_sl_matrix = _transform_points_to_make_one_infinity_and_inv_sl_matrix(idealPoints)
    transformedInCenter = _compute_incenter_of_triangle_with_one_point_at_infinity(transformedIdealPoints)
    return _translate(transformedInCenter, inv_sl_matrix)