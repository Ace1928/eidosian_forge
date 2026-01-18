from .sage_helper import _within_sage
from . import number
from .math_basics import is_Interval
def mat_solve(m, v, epsilon=0):
    """
    Given a matrix m and a vector v, return the vector a such that
    v = m * a - computed using Gaussian elimination.

    Note that the matrix and vector can contain real or complex numbers
    or intervals (SageMath's RealIntervalField, ComplexIntervalField).

    When not given intervals, an epsilon can be specified. If a pivot
    has absolute value less than the given epsilon, a ZeroDivisionError
    will be raised indicating that the matrix is degenerate.

    We provide mat_solve for two reasons:
      1. To have this functionality outside of SageMath.
      2. To avoid bad numerical results even though the matrix is far
         from degenerate, it is necessary to swap rows during elimination
         when the pivot is really small. However, SageMath instead checks
         whether the pivot is exactly zero rather than close to zero for
         some numerical types. In particular, this applies to interval
         types and SageMath often returns matrices with entries (-inf, inf)
         even though the matrix is far from degenerate.

    Our implementation improves on this by swapping rows so that the
    element with the largest (lower bound of the) absolute value is
    used as pivot.

    Setup a complex interval for example::

        sage: from sage.all import RealIntervalField, ComplexIntervalField
        sage: RIF = RealIntervalField(80)
        sage: CIF = ComplexIntervalField(80)
        sage: fuzzy_four = CIF(RIF(3.9999,4.0001),RIF(-0.0001,0.0001))

    Construct a matrix/vector with complex interval coefficients. One entry
    is a complex interval with non-zero diameter::

        sage: m = matrix(CIF,
        ...      [  [ fuzzy_four, 3, 2, 3],
        ...         [          2, 3, 6, 2],
        ...         [          2, 4, 1, 6],
        ...         [          3, 2,-5, 2]])
        sage: v = vector(CIF, [fuzzy_four, 2, 0 ,1])

    Now compute the solutions a to v = m * a::

        sage: a = mat_solve(m, v)
        sage: a  # doctest: +ELLIPSIS
        (1.5...? + 0.000?*I, -1.2...? + 0.000?*I, 0.34...? + 0.0000?*I, 0.24...? + 0.000?*I)
        sage: m * a  # doctest: +ELLIPSIS
        (4.0...? + 0.00?*I, 2.0...? + 0.00?*I, 0.0...? + 0.00?*I, 1.00? + 0.00?*I)

    The product actually contains the vector v, we check entry wise::

        sage: [s in t for s, t in zip(v, m * a)]
        [True, True, True, True]
    """
    dim0, dim1 = m.dimensions()
    if dim0 != len(v) or dim1 != len(v):
        raise ValueError('mat_solve was given a vector with length %d not matching the size %dx%d of the matrix.' % (len(v), dim0, dim1))
    is_interval = is_Interval(m[0, 0])
    if is_interval and (not epsilon == 0):
        raise ValueError("mat_solve's epsilon has to be exactly 0 for verified computations with intervals.")
    m1 = [[m[i][j] for j in range(dim0)] + [v[i]] for i in range(dim0)]
    for i in range(dim0):
        if is_interval:
            pivots = [(j, m1[j][i].abs().lower()) for j in range(i, dim0)]
        else:
            pivots = [(j, m1[j][i].abs()) for j in range(i, dim0)]
        max_index, max_val = max(pivots, key=lambda x: x[1])
        if not max_val > epsilon:
            raise ZeroDivisionError
        if max_index != i:
            for j in range(i, dim0 + 1):
                m1[max_index][j], m1[i][j] = (m1[i][j], m1[max_index][j])
        for j in range(i + 1, dim0 + 1):
            m1[i][j] /= m1[i][i]
        for j in range(dim0):
            if i != j:
                for k in range(i + 1, dim0 + 1):
                    m1[j][k] -= m1[j][i] * m1[i][k]
    return vector([row[-1] for row in m1])