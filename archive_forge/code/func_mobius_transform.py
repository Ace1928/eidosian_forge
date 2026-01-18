from sympy.core import S, Symbol, sympify
from sympy.core.function import expand_mul
from sympy.core.numbers import pi, I
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.ntheory import isprime, primitive_root
from sympy.utilities.iterables import ibin, iterable
from sympy.utilities.misc import as_int
def mobius_transform(seq, subset=True):
    """
    Performs the Mobius Transform for subset lattice with indices of
    sequence as bitmasks.

    The indices of each argument, considered as bit strings, correspond
    to subsets of a finite set.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset/superset based on bitmasks (indices) requires
    the size of sequence to be a power of 2.

    Parameters
    ==========

    seq : iterable
        The sequence on which Mobius Transform is to be applied.
    subset : bool
        Specifies if Mobius Transform is applied by enumerating subsets
        or supersets of the given set.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy import mobius_transform, inverse_mobius_transform
    >>> x, y, z = symbols('x y z')

    >>> mobius_transform([x, y, z])
    [x, x + y, x + z, x + y + z]
    >>> inverse_mobius_transform(_)
    [x, y, z, 0]

    >>> mobius_transform([x, y, z], subset=False)
    [x + y + z, y, z, 0]
    >>> inverse_mobius_transform(_, subset=False)
    [x, y, z, 0]

    >>> mobius_transform([1, 2, 3, 4])
    [1, 3, 4, 10]
    >>> inverse_mobius_transform(_)
    [1, 2, 3, 4]
    >>> mobius_transform([1, 2, 3, 4], subset=False)
    [10, 6, 7, 4]
    >>> inverse_mobius_transform(_, subset=False)
    [1, 2, 3, 4]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_inversion_formula
    .. [2] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf
    .. [3] https://arxiv.org/pdf/1211.0189.pdf

    """
    return _mobius_transform(seq, sgn=+1, subset=subset)