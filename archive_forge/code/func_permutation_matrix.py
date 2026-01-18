from sympy.core.backend import Matrix, eye, zeros
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import flatten
from sympy.physics.vector import dynamicsymbols
from sympy.physics.mechanics.functions import msubs
from collections import namedtuple
from collections.abc import Iterable
def permutation_matrix(orig_vec, per_vec):
    """Compute the permutation matrix to change order of
    orig_vec into order of per_vec.

    Parameters
    ==========

    orig_vec : array_like
        Symbols in original ordering.
    per_vec : array_like
        Symbols in new ordering.

    Returns
    =======

    p_matrix : Matrix
        Permutation matrix such that orig_vec == (p_matrix * per_vec).
    """
    if not isinstance(orig_vec, (list, tuple)):
        orig_vec = flatten(orig_vec)
    if not isinstance(per_vec, (list, tuple)):
        per_vec = flatten(per_vec)
    if set(orig_vec) != set(per_vec):
        raise ValueError('orig_vec and per_vec must be the same length, ' + 'and contain the same symbols.')
    ind_list = [orig_vec.index(i) for i in per_vec]
    p_matrix = zeros(len(orig_vec))
    for i, j in enumerate(ind_list):
        p_matrix[i, j] = 1
    return p_matrix