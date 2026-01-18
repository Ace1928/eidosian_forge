from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
def test_indent_unknown_type_acceptance(self):
    """
        A test against the regression mentioned at `github issue 29`_.

        The indent parameter should accept any type which pretends to be
        an instance of int or long when it comes to being multiplied by
        strings, even if it is not actually an int or long, for
        backwards compatibility.

        .. _github issue 29:
           http://github.com/simplejson/simplejson/issue/29
        """

    class AwesomeInt(object):
        """An awesome reimplementation of integers"""

        def __init__(self, *args, **kwargs):
            if len(args) > 0:
                if isinstance(args[0], int):
                    self._int = args[0]

        def __mul__(self, other):
            if hasattr(self, '_int'):
                return self._int * other
            else:
                raise NotImplementedError('To do non-awesome things with this object, please construct it from an integer!')
    s = json.dumps([0, 1, 2], indent=AwesomeInt(3))
    self.assertEqual(s, '[\n   0,\n   1,\n   2\n]')