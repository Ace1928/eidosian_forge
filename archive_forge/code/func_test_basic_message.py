import pyomo.common.unittest as unittest
from pyomo.common.errors import format_exception
def test_basic_message(self):
    self.assertEqual(format_exception('Hello world'), 'Hello world')