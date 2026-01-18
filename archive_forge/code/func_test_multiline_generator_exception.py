from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
def test_multiline_generator_exception(self):
    os = StringIO()
    data = {'a': 0, 'b': 1, 'c': 3}

    def _data_gen(i, j):
        if i == 'b':
            raise ValueError('invalid')
        for n in range(j):
            yield (n, chr(ord('a') + n) * j)
    tabular_writer(os, '', data.items(), ['i', 'j'], _data_gen)
    ref = u'\nKey : i    : j\n  a : None : None\n  b : None : None\n  c :    0 :  aaa\n    :    1 :  bbb\n    :    2 :  ccc\n'
    self.assertEqual(ref.strip(), os.getvalue().strip())