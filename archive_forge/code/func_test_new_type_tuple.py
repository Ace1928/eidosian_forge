from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
def test_new_type_tuple(self):
    self.assertEqual(tostr(DerivedTuple([1, 2])), '(1, 2)')
    self.assertIs(tostr.handlers[DerivedTuple], tostr.handlers[tuple])