from sympy.core import S, sympify
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int
def linrec_coeffs(c, n):
    """
    Compute the coefficients of n'th term in linear recursion
    sequence defined by c.

    `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \\cdots + c_{k-1}`.

    It computes the coefficients by using binary exponentiation.
    This function is used by `linrec` and `_eval_pow_by_cayley`.

    Parameters
    ==========

    c = coefficients of the divisor polynomial
    n = exponent of x, so dividend is x^n

    """
    k = len(c)

    def _square_and_reduce(u, offset):
        w = [S.Zero] * (2 * len(u) - 1 + offset)
        for i, p in enumerate(u):
            for j, q in enumerate(u):
                w[offset + i + j] += p * q
        for j in range(len(w) - 1, k - 1, -1):
            for i in range(k):
                w[j - i - 1] += w[j] * c[i]
        return w[:k]

    def _final_coeffs(n):
        if n < k:
            return [S.Zero] * n + [S.One] + [S.Zero] * (k - n - 1)
        else:
            return _square_and_reduce(_final_coeffs(n // 2), n % 2)
    return _final_coeffs(n)