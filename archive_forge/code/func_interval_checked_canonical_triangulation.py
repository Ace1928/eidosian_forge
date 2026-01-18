from ..sage_helper import _within_sage, sage_method
from .cuspCrossSection import RealCuspCrossSection
from .squareExtensions import find_shapes_as_complex_sqrt_lin_combinations
from . import verifyHyperbolicity
from . import exceptions
from ..exceptions import SnapPeaFatalError
from ..snap import t3mlite as t3m
@sage_method
def interval_checked_canonical_triangulation(M, bits_prec=None):
    """
    Given a canonical triangulation of a cusped (possibly non-orientable)
    manifold M, return this triangulation if it has tetrahedral cells and can
    be verified using interval arithmetics with the optional, given precision.
    Otherwise, raises an Exception.

    It fails when we call it on something which is not the canonical
    triangulation::

       sage: from snappy import Manifold
       sage: M = Manifold("m015")
       sage: interval_checked_canonical_triangulation(M) # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
       Traceback (most recent call last):
       ...
       TiltProvenPositiveNumericalVerifyError: Numerical verification that tilt is negative has failed, tilt is actually positive. This is provably not the proto-canonical triangulation: 0.164542163...? <= 0

    It verifies the canonical triangulation::

       sage: M.canonize()
       sage: K = interval_checked_canonical_triangulation(M)
       sage: K
       m015(0,0)

    Has a non-tetrahedral canonical cell::

      sage: M = Manifold("m137")
      sage: M.canonize()
      sage: interval_checked_canonical_triangulation(M) # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
      ...
      TiltInequalityNumericalVerifyError: Numerical verification that tilt is negative has failed: 0.?e-1... < 0

    Has a cubical canonical cell::

       sage: M = Manifold("m412")
       sage: M.canonize()
       sage: interval_checked_canonical_triangulation(M) # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
       Traceback (most recent call last):
       ...
       TiltInequalityNumericalVerifyError: Numerical verification that tilt is negative has failed: 0.?e-1... < 0

    """
    shapes = M.tetrahedra_shapes('rect', intervals=True, bits_prec=bits_prec)
    c = RealCuspCrossSection.fromManifoldAndShapes(M, shapes)
    verifyHyperbolicity.check_logarithmic_gluing_equations_and_positively_oriented_tets(M, shapes)
    if M.num_cusps() > 1:
        c.normalize_cusps()
    c.compute_tilts()
    for face in c.mcomplex.Faces:
        if face.Tilt > 0:
            raise exceptions.TiltProvenPositiveNumericalVerifyError(face.Tilt)
        if not face.Tilt < 0:
            raise exceptions.TiltInequalityNumericalVerifyError(face.Tilt)
    return M