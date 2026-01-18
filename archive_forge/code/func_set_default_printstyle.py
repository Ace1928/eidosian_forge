from .polynomial import Polynomial
from .chebyshev import Chebyshev
from .legendre import Legendre
from .hermite import Hermite
from .hermite_e import HermiteE
from .laguerre import Laguerre
from numpy._pytesttester import PytestTester
def set_default_printstyle(style):
    """
    Set the default format for the string representation of polynomials.

    Values for ``style`` must be valid inputs to ``__format__``, i.e. 'ascii'
    or 'unicode'.

    Parameters
    ----------
    style : str
        Format string for default printing style. Must be either 'ascii' or
        'unicode'.

    Notes
    -----
    The default format depends on the platform: 'unicode' is used on
    Unix-based systems and 'ascii' on Windows. This determination is based on
    default font support for the unicode superscript and subscript ranges.

    Examples
    --------
    >>> p = np.polynomial.Polynomial([1, 2, 3])
    >>> c = np.polynomial.Chebyshev([1, 2, 3])
    >>> np.polynomial.set_default_printstyle('unicode')
    >>> print(p)
    1.0 + 2.0·x + 3.0·x²
    >>> print(c)
    1.0 + 2.0·T₁(x) + 3.0·T₂(x)
    >>> np.polynomial.set_default_printstyle('ascii')
    >>> print(p)
    1.0 + 2.0 x + 3.0 x**2
    >>> print(c)
    1.0 + 2.0 T_1(x) + 3.0 T_2(x)
    >>> # Formatting supersedes all class/package-level defaults
    >>> print(f"{p:unicode}")
    1.0 + 2.0·x + 3.0·x²
    """
    if style not in ('unicode', 'ascii'):
        raise ValueError(f"Unsupported format string '{style}'. Valid options are 'ascii' and 'unicode'")
    _use_unicode = True
    if style == 'ascii':
        _use_unicode = False
    from ._polybase import ABCPolyBase
    ABCPolyBase._use_unicode = _use_unicode