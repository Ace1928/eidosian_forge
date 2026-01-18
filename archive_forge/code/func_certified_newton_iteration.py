from ..matrix import matrix, vector, mat_solve
from .. import snap
from ..sage_helper import _within_sage, sage_method
@staticmethod
def certified_newton_iteration(equations, shape_intervals, point_in_intervals=None, interval_value_at_point=None):
    """
        Given shape intervals z, performs a Newton interval iteration N(z)
        as described in newton_iteration. Returns a pair (boolean, N(z)) where
        the boolean is True if N(z) is contained in z.

        If the boolean is True, it is certified that N(z) contains a true
        solution, e.g., a point for which f is truly zero.

        See newton_iteration for the other parameters.

        This follows from Theorem 1 of `Zgliczynski's notes
        <http://ww2.ii.uj.edu.pl/~zgliczyn/cap07/krawczyk.pdf>`_.

        Some examples::

            sage: from snappy import Manifold
            sage: M = Manifold("m019")
            sage: C = IntervalNewtonShapesEngine(M, M.tetrahedra_shapes('rect'),
            ...                           bits_prec = 80)

        Intervals containing the true solution::

            sage: good_shapes = vector([
            ...       C.CIF(C.RIF(0.78055, 0.78056), C.RIF(0.91447, 0.91448)),
            ...       C.CIF(C.RIF(0.78055, 0.78056), C.RIF(0.91447, 0.91448)),
            ...       C.CIF(C.RIF(0.46002, 0.46003), C.RIF(0.63262, 0.63263))])
            sage: is_certified, shapes = IntervalNewtonShapesEngine.certified_newton_iteration(C.equations, good_shapes)

            sage: is_certified
            True
            sage: shapes  # doctest: +ELLIPSIS
            (0.78055253? + 0.91447366...?*I, 0.7805525...? + 0.9144736...?*I, 0.4600211...? + 0.632624...?*I)

        This means that a true solution to the rectangular gluing equations is
        contained in both the given intervals (good_shapes) and the returned
        intervals (shapes) which are a refinement of the given intervals.

        Intervals not containing a true solution::

            sage: bad_shapes = vector([
            ...       C.CIF(C.RIF(0.78054, 0.78055), C.RIF(0.91447, 0.91448)),
            ...       C.CIF(C.RIF(0.78055, 0.78056), C.RIF(0.91447, 0.91448)),
            ...       C.CIF(C.RIF(0.46002, 0.46003), C.RIF(0.63262, 0.63263))])
            sage: is_certified, shapes = IntervalNewtonShapesEngine.certified_newton_iteration(C.equations, bad_shapes)
            sage: is_certified
            False

        """
    new_shapes = IntervalNewtonShapesEngine.newton_iteration(equations, shape_intervals, point_in_intervals=point_in_intervals, interval_value_at_point=interval_value_at_point)
    return (IntervalNewtonShapesEngine.interval_vector_is_contained_in(new_shapes, shape_intervals), new_shapes)