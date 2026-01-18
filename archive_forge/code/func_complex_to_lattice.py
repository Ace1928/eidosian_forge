from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
def complex_to_lattice(z, d, a, N=None):
    """
    Given an algebraic number z of degree d, set of up the
    lattice which tries to express a in terms of powers of z,
    where the last two columns are weighted by N.
    """
    if N is None:
        N = ZZ(2) ** (z.prec() - 10)
    nums = [z ** k for k in range(d)] + [a]
    last_columns = [[round(N * real_part(x)) for x in nums], [round(N * imag_part(x)) for x in nums]]
    A = matrix(identity_matrix(len(nums)).columns() + last_columns)
    return A.transpose()