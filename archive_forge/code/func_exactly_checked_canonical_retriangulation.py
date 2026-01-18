from ..sage_helper import _within_sage, sage_method
from .cuspCrossSection import RealCuspCrossSection
from .squareExtensions import find_shapes_as_complex_sqrt_lin_combinations
from . import verifyHyperbolicity
from . import exceptions
from ..exceptions import SnapPeaFatalError
from ..snap import t3mlite as t3m
@sage_method
def exactly_checked_canonical_retriangulation(M, bits_prec, degree):
    """
    Given a proto-canonical triangulation of a cusped (possibly non-orientable)
    manifold M, return its canonical retriangulation which is computed from
    exact shapes. The exact shapes are computed using snap (which uses the
    LLL-algorithm). The precision (in bits) and the maximal degree need to be
    specified (here 300 bits precision and polynomials of degree less than 4)::

       sage: from snappy import Manifold
       sage: M = Manifold("m412")
       sage: M.canonize()
       sage: K = exactly_checked_canonical_retriangulation(M, 300, 4)

    M's canonical cell decomposition has a cube, so non-tetrahedral::

       sage: K.has_finite_vertices()
       True

    Has 12 tetrahedra after the retrianglation::

      sage: K.num_tetrahedra()
      12

    Check that it fails on something which is not a proto-canonical
    triangulation::

      sage: from snappy import Manifold
      sage: M = Manifold("m015")
      sage: exactly_checked_canonical_retriangulation(M, 500, 6)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
      ...
      TiltProvenPositiveNumericalVerifyError: Numerical verification that tilt is negative has failed, tilt is actually positive. This is provably not the proto-canonical triangulation: 0.1645421638874662848910671879? <= 0
    """
    dec_prec = prec_bits_to_dec(bits_prec)
    shapes = find_shapes_as_complex_sqrt_lin_combinations(M, dec_prec, degree)
    if not shapes:
        raise FindExactShapesError()
    c = RealCuspCrossSection.fromManifoldAndShapes(M, shapes)
    c.check_polynomial_edge_equations_exactly()
    c.check_cusp_development_exactly()
    CIF = ComplexIntervalField(bits_prec)
    c.check_logarithmic_edge_equations_and_positivity(CIF)
    if M.num_cusps() > 1:
        c.normalize_cusps()
    c.compute_tilts()

    def get_opacity(tilt):
        sign, interval = tilt.sign_with_interval()
        if sign < 0:
            return True
        if sign == 0:
            return False
        if sign > 0:
            raise exceptions.TiltProvenPositiveNumericalVerifyError(interval)

    def index_of_face_corner(corner):
        face_index = t3m.simplex.comp(corner.Subsimplex).bit_length() - 1
        return 4 * corner.Tetrahedron.Index + face_index
    opacities = 4 * len(c.mcomplex.Tetrahedra) * [None]
    for face in c.mcomplex.Faces:
        opacity = get_opacity(face.Tilt)
        for corner in face.Corners:
            opacities[index_of_face_corner(corner)] = opacity
    if None in opacities:
        raise Exception('Mismatch with opacities')
    if False in opacities:
        return M._canonical_retriangulation(opacities)
    return M