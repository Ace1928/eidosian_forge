import pyomo.common.unittest as unittest
from pyomo.common.errors import format_exception
def test_long_basic_message_builtin_exception(self):
    self.assertEqual(format_exception('Hello world, this is a very long message that will inevitably wrap onto another line.', exception=RuntimeError), 'Hello world, this is a very long message that will inevitably\n    wrap onto another line.')