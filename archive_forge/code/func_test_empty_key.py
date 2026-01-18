import unittest
from oslo_config import iniparser
def test_empty_key(self):
    self._assertParseError(': bar')