import unittest
from oslo_config import iniparser
def test_assignment_space_single_quote(self):
    lines = ["foo = ' bar '"]
    self.parser.parse(lines)
    self.assertEqual({'': {'foo': [' bar ']}}, self.parser.values)