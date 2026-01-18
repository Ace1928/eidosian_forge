from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
def test_multiline_alignment(self):
    os = StringIO()
    data = {'a': 1, 'b': 2, 'c': 3}

    def _data_gen(i, j):
        for n in range(j):
            _str = chr(ord('a') + n) * (j + 1)
            if n % 2:
                _str = list(_str)
                _str[1] = ' '
                _str = ''.join(_str)
            yield (n, _str)
    tabular_writer(os, '', data.items(), ['i', 'j'], _data_gen)
    ref = u'\nKey : i : j\n  a : 0 : aa\n  b : 0 : aaa\n    : 1 : b b\n  c : 0 : aaaa\n    : 1 : b bb\n    : 2 : cccc\n'
    self.assertEqual(ref.strip(), os.getvalue().strip())