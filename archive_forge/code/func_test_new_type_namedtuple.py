from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
def test_new_type_namedtuple(self):
    self.assertEqual(tostr(NamedTuple(1, 2)), 'NamedTuple(x=1, y=2)')
    self.assertIs(tostr.handlers[NamedTuple], tostr.handlers[None])