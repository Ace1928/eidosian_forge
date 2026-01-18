import unittest
from cliff import columns
def test_machine_readable(self):
    c = FauxColumn(['list', 'of', 'values'])
    self.assertEqual(['list', 'of', 'values'], c.machine_readable())