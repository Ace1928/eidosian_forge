from numbers import Number
from future.utils import PY3, istext, with_metaclass, isnewbytes
from future.types import no, issubset
from future.types.newobject import newobject
@staticmethod
def maketrans(x, y=None, z=None):
    """
        Return a translation table usable for str.translate().

        If there is only one argument, it must be a dictionary mapping Unicode
        ordinals (integers) or characters to Unicode ordinals, strings or None.
        Character keys will be then converted to ordinals.
        If there are two arguments, they must be strings of equal length, and
        in the resulting dictionary, each character in x will be mapped to the
        character at the same position in y. If there is a third argument, it
        must be a string, whose characters will be mapped to None in the result.
        """
    if y is None:
        assert z is None
        if not isinstance(x, dict):
            raise TypeError('if you give only one argument to maketrans it must be a dict')
        result = {}
        for key, value in x.items():
            if len(key) > 1:
                raise ValueError('keys in translate table must be strings or integers')
            result[ord(key)] = value
    else:
        if not isinstance(x, unicode) and isinstance(y, unicode):
            raise TypeError('x and y must be unicode strings')
        if not len(x) == len(y):
            raise ValueError('the first two maketrans arguments must have equal length')
        result = {}
        for xi, yi in zip(x, y):
            if len(xi) > 1:
                raise ValueError('keys in translate table must be strings or integers')
            result[ord(xi)] = ord(yi)
    if z is not None:
        for char in z:
            result[ord(char)] = None
    return result