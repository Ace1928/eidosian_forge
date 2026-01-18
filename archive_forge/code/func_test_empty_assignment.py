import unittest
from oslo_config import iniparser
def test_empty_assignment(self):
    lines = ['foo = ']
    self.parser.parse(lines)
    self.assertEqual({'': {'foo': ['']}}, self.parser.values)