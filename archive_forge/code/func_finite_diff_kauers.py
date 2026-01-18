def finite_diff_kauers(sum):
    """
    Takes as input a Sum instance and returns the difference between the sum
    with the upper index incremented by 1 and the original sum. For example,
    if S(n) is a sum, then finite_diff_kauers will return S(n + 1) - S(n).

    Examples
    ========

    >>> from sympy.series.kauers import finite_diff_kauers
    >>> from sympy import Sum
    >>> from sympy.abc import x, y, m, n, k
    >>> finite_diff_kauers(Sum(k, (k, 1, n)))
    n + 1
    >>> finite_diff_kauers(Sum(1/k, (k, 1, n)))
    1/(n + 1)
    >>> finite_diff_kauers(Sum((x*y**2), (x, 1, n), (y, 1, m)))
    (m + 1)**2*(n + 1)
    >>> finite_diff_kauers(Sum((x*y), (x, 1, m), (y, 1, n)))
    (m + 1)*(n + 1)
    """
    function = sum.function
    for l in sum.limits:
        function = function.subs(l[0], l[-1] + 1)
    return function