from ..matrix import matrix, vector, mat_solve
from .. import snap
from ..sage_helper import _within_sage, sage_method
@staticmethod
def newton_iteration(equations, shape_intervals, point_in_intervals=None, interval_value_at_point=None):
    """
        Perform a Newton interval method of iteration for
        the function f described in log_gluing_LHSs.

        Let z denote the shape intervals.
        Let z_center be a point close to the center point of the shape
        intervals (in the implementation, z_center is an interval of
        again, of length zero).

        The result returned will be

                    N(z) = z_center - ((Df)(z))^-1 f(z_center)

        The user can overwrite the z_center to be used by providing
        point_in_intervals (which have to be 0-length complex intervals).
        The user can also give the interval value of f(z_center) by providing
        interval_value_at_point to avoid re-evaluation of f(z_center).

        A very approximate solution::

            sage: from snappy import Manifold
            sage: M = Manifold("m019")
            sage: shapes = [ 0.7+1j, 0.7+1j, 0.5+0.5j ]

        Get the equations and initialize zero-length intervals from it::

            sage: C = IntervalNewtonShapesEngine(M, shapes, bits_prec = 80)
            sage: C.initial_shapes
            (0.69999999999999995559107902? + 1*I, 0.69999999999999995559107902? + 1*I, 0.50000000000000000000000000? + 0.50000000000000000000000000?*I)

        Do several Newton interval operations to get a better solution::

            sage: shape_intervals = C.initial_shapes
            sage: for i in range(4): # doctest: +ELLIPSIS
            ...     shape_intervals = IntervalNewtonShapesEngine.newton_iteration(C.equations, shape_intervals)
            ...     print(shape_intervals)
            (0.78674683118381457770...? + 0.9208680745160821379529?*I, 0.786746831183814577703...? + 0.9208680745160821379529?*I, 0.459868058287098030934...? + 0.61940871855835167317...?*I)
            (0.78056102517632648594...? + 0.9144962118446750482...?*I, 0.78056102517632648594...? + 0.9144962118446750482...?*I, 0.4599773577869384936554? + 0.63251940718694538695...?*I)
            (0.78055253104531610049...? + 0.9144736621585220345231?*I, 0.780552531045316100497...? + 0.9144736621585220345231?*I, 0.460021167103732494700...? + 0.6326241909236695020810...?*I)
            (0.78055252785072483256...? + 0.91447366296772644033...?*I, 0.7805525278507248325678? + 0.914473662967726440333...?*I, 0.4600211755737178641204...? + 0.6326241936052562241142...?*I)

        For comparison::

            sage: M.tetrahedra_shapes('rect')
            [0.780552527850725 + 0.914473662967726*I, 0.780552527850725 + 0.914473662967726*I, 0.460021175573718 + 0.632624193605256*I]

        Start with a rather big interval, note that the Newton interval method is
        stable in the sense that the interval size decreases::

            sage: box = C.CIF(C.RIF(-0.0001,0.0001),C.RIF(-0.0001,0.0001))
            sage: shape_intervals = C.initial_shapes.apply_map(lambda shape: shape + box)
            sage: shape_intervals
            (0.700? + 1.000?*I, 0.700? + 1.000?*I, 0.500? + 0.500?*I)
            sage: for i in range(7):
            ...     shape_intervals = IntervalNewtonShapesEngine.newton_iteration(C.equations, shape_intervals)
            sage: print(shape_intervals) # doctest: +ELLIPSIS
            (0.78055252785072483798...? + 0.91447366296772645593...?*I, 0.7805525278507248379869? + 0.914473662967726455938...?*I, 0.460021175573717872891...? + 0.632624193605256171637...?*I)


        """
    if point_in_intervals is None:
        point_in_intervals = IntervalNewtonShapesEngine.interval_vector_mid_points(shape_intervals)
    if interval_value_at_point is None:
        interval_value_at_point = IntervalNewtonShapesEngine.log_gluing_LHSs(equations, point_in_intervals)
    derivatives = IntervalNewtonShapesEngine.log_gluing_LHS_derivatives(equations, shape_intervals)
    return point_in_intervals - mat_solve(derivatives, interval_value_at_point)