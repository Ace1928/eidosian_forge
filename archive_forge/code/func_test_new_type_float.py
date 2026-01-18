from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
def test_new_type_float(self):
    self.assertEqual(tostr(0.5), '0.5')
    self.assertIs(tostr.handlers[float], tostr.handlers[None])