from ..sage_helper import _within_sage, sage_method
from .. import snap
from . import exceptions
@sage_method
def check_logarithmic_gluing_equations_and_positively_oriented_tets(manifold, shape_intervals):
    """
    Given a SnapPy manifold manifold and complex intervals for the shapes
    shape_intervals that are certified to contain a solution to the
    rectangular gluing equations, verify that the logarithmic gluing equations
    are also fulfilled and that all shapes have positive imaginary part.
    It will raise an exception if the verification fails.
    This is sufficient to prove that the manifold is indeed hyperbolic.

    Since the given interval are supposed to contain a true solution of
    the rectangular gluing equations, the logarithmic gluing equations
    are known to be fulfilled up to a multiple of 2 pi i. Thus it is enough
    to certify that the  absolute error of the logarithmic gluing
    equations is < 0.1. Using interval arithmetic, this function certifies
    this and positivity of the imaginary parts of the shapes::

        sage: from snappy import Manifold
        sage: M = Manifold("m019")
        sage: check_logarithmic_gluing_equations_and_positively_oriented_tets(
        ...    M, M.tetrahedra_shapes('rect', intervals=True))


    The SnapPy triangulation of the following hyperbolic manifold contains
    actually negatively oriented tetrahedra::

        sage: M = Manifold("t02774")
        sage: check_logarithmic_gluing_equations_and_positively_oriented_tets(
        ...    M, M.tetrahedra_shapes('rect', intervals=True))    # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ShapePositiveImaginaryPartNumericalVerifyError: Numerical verification that shape has positive imaginary part has failed: Im(0.4800996900657? - 0.0019533695046?*I) > 0


    """
    for d in manifold.cusp_info():
        m, l = d['filling']
        if not (m.is_integer() and l.is_integer()):
            raise NonIntegralFillingsError(M)
    for shape in shape_intervals:
        if not shape.imag() > 0:
            raise exceptions.ShapePositiveImaginaryPartNumericalVerifyError(shape)
    logZ = [z.log() for z in shape_intervals]
    logZp = [(1 / (1 - z)).log() for z in shape_intervals]
    logZpp = [((z - 1) / z).log() for z in shape_intervals]
    logs = [z for triple in zip(logZ, logZp, logZpp) for z in triple]
    n_tet = manifold.num_tetrahedra()
    n_cusps = manifold.num_cusps()
    equations = manifold.gluing_equations()
    LHSs = [sum([l * expo for l, expo in zip(logs, equation)]) for equation in equations]
    CIF = shape_intervals[0].parent()
    RIF = CIF.real_field()
    two_pi_i = CIF(2 * pi * sage.all.I)
    LHS_index = 0
    for edge_index in range(n_tet):
        if not abs(LHSs[LHS_index] - two_pi_i) < RIF(0.1):
            raise exceptions.EdgeEquationLogLiftNumericalVerifyError(LHSs[LHS_index])
        LHS_index += 1
    for cusp_index in range(n_cusps):
        num_LHSs, value = (2, 0) if manifold.cusp_info(cusp_index)['complete?'] else (1, two_pi_i)
        for j in range(num_LHSs):
            if not abs(LHSs[LHS_index] - value) < RIF(0.1):
                raise exceptions.CuspEquationLogLiftNumericalVerifyError(LHSs[LHS_index], value)
            LHS_index += 1