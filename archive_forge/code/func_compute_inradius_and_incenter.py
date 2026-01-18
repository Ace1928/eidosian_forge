from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def compute_inradius_and_incenter(idealPoints):
    """
    Computes inradius and incenter of the tetrahedron spanned by four
    ideal points::

        sage: from sage.all import CIF
        sage: z0 = Infinity
        sage: z1 = CIF(0)
        sage: z2 = CIF(1)
        sage: z3 = CIF(1.2, 1.0)
        sage: compute_inradius_and_incenter([z0, z1, z2, z3])
        (0.29186158033099?, FinitePoint(0.771123016231387? + 0.2791850380434060?*I, 0.94311979279000?))
    """
    if not len(idealPoints) == 4:
        raise Exception('Expected 4 ideal points.')
    transformedIdealPoints, inv_sl_matrix = _transform_points_to_make_one_infinity_and_inv_sl_matrix(idealPoints)
    inradius, transformedInCenter = _compute_inradius_and_incenter_with_one_point_at_infinity(transformedIdealPoints)
    return (inradius, _translate(transformedInCenter, inv_sl_matrix))