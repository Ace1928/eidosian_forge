import unittest
from oslo_config import iniparser
def test_invalid_section(self):
    self._assertParseError('[section')