from snappy import snap
from snappy.sage_helper import _within_sage, sage_method
def log_gluing_LHSs(self, shapes):
    """
        Given the result of M.gluing_equations('rect') or a
        subset of rows of it and shapes, return a vector of
        log(LHS) where

           LHS = c * z0 ** a0 * (1-z0) ** b0 * z1 ** a1 * ...

        Let f: C^n -> C^n denote the function which takes
        shapes and returns the vector of log(LHS).

        The reason we take the logarithm of the rectangular
        gluing equations is because the logarithmic derivative
        is of a particular nice form::

            sage: from snappy import Manifold
            sage: M = Manifold("m019")
            sage: equations = M.gluing_equations('rect')
            sage: RIF = RealIntervalField(80)
            sage: CIF = ComplexIntervalField(80)
            sage: zero = CIF(0).center()
            sage: shape1 = CIF(RIF(0.78055,0.78056), RIF(0.9144, 0.9145))
            sage: shape2 = CIF(RIF(0.46002,0.46003), RIF(0.6326, 0.6327))

        An interval solution containing the true solution. The log of each
        rectangular equation should be 0 for the true solution, hence the interval
        should contain zero::

            sage: shapes = [shape1, shape1, shape2]
            sage: C = KrawczykShapesEngine(M, [shape.center() for shape in shapes], bits_prec = 53)
            sage: LHSs = C.log_gluing_LHSs(shapes)
            sage: LHSs # doctest: +NUMERIC6
            (0.000? + 0.000?*I, 0.000? + 0.000?*I, 0.0000? + 0.0000?*I)
            sage: zero in LHSs[0]
            True

        An interval not containing the true solution::

            sage: shapes = [shape1, shape1, shape1]
            sage: LHSs = C.log_gluing_LHSs(shapes)
            sage: LHSs # doctest: +NUMERIC3
            (0.430? - 0.078?*I, 0.246? - 0.942?*I, 0.0000? + 0.0000?*I)
            sage: zero in LHSs[0]
            False

        """
    BaseField = shapes[0].parent()
    one = BaseField(1)
    gluing_LHSs = []
    for A, B, c in self.equations:
        prod = BaseField(c)
        for a, b, shape in zip(A, B, shapes):
            prod *= shape ** a * (one - shape) ** b
        gluing_LHSs.append(prod.log())
    return vector(BaseField, gluing_LHSs)