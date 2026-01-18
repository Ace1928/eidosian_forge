from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
def test_blank_lines(self):
    OUT1 = StringIO()
    OUT2 = StreamIndenter(OUT1)
    OUT2.write('Hello?\n\nText\n\nHello, world!')
    self.assertEqual('    Hello?\n\n    Text\n\n    Hello, world!', OUT2.getvalue())