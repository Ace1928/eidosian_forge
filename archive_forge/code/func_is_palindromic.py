from collections import defaultdict
from sympy.utilities.iterables import multiset, is_palindromic as _palindromic
from sympy.utilities.misc import as_int
def is_palindromic(n, b=10):
    """return True if ``n`` is the same when read from left to right
    or right to left in the given base, ``b``.

    Examples
    ========

    >>> from sympy.ntheory import is_palindromic

    >>> all(is_palindromic(i) for i in (-11, 1, 22, 121))
    True

    The second argument allows you to test numbers in other
    bases. For example, 88 is palindromic in base-10 but not
    in base-8:

    >>> is_palindromic(88, 8)
    False

    On the other hand, a number can be palindromic in base-8 but
    not in base-10:

    >>> 0o121, is_palindromic(0o121)
    (81, False)

    Or it might be palindromic in both bases:

    >>> oct(121), is_palindromic(121, 8) and is_palindromic(121)
    ('0o171', True)

    """
    return _palindromic(digits(n, b), 1)