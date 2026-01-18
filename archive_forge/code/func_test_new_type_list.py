from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
def test_new_type_list(self):
    self.assertEqual(tostr(DerivedList([1, 2])), '[1, 2]')
    self.assertIs(tostr.handlers[DerivedList], tostr.handlers[list])