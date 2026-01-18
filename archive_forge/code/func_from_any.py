from numbers import Real
from matplotlib import _api
from matplotlib.axes import Axes
def from_any(size, fraction_ref=None):
    """
    Create a Fixed unit when the first argument is a float, or a
    Fraction unit if that is a string that ends with %. The second
    argument is only meaningful when Fraction unit is created.

    >>> from mpl_toolkits.axes_grid1.axes_size import from_any
    >>> a = from_any(1.2) # => Fixed(1.2)
    >>> from_any("50%", a) # => Fraction(0.5, a)
    """
    if isinstance(size, Real):
        return Fixed(size)
    elif isinstance(size, str):
        if size[-1] == '%':
            return Fraction(float(size[:-1]) / 100, fraction_ref)
    raise ValueError('Unknown format')