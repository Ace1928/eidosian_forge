from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix
def images_have_same_distance(self, m):
    from sage.rings.real_mpfi import RealIntervalFieldElement
    from sage.all import RIF, CIF
    a = FinitePoint(CIF(RIF(3.5), RIF(-3.0)), RIF(8.5))
    b = FinitePoint(CIF(RIF(4.5), RIF(-4.5)), RIF(9.6))
    d_before = a.dist(b)
    a = a.translate_PGL(m)
    b = b.translate_PGL(m)
    d_after = a.dist(b)
    if not isinstance(d_before, RealIntervalFieldElement):
        raise Exception('Expected distance to be RIF')
    if not isinstance(d_after, RealIntervalFieldElement):
        raise Exception('Expected distance to be RIF')
    if not abs(d_before - d_after) < RIF(1e-12):
        raise Exception('Distance changed %r %r' % (d_before, d_after))