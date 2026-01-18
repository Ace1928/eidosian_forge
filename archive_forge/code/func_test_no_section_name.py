import unittest
from oslo_config import iniparser
def test_no_section_name(self):
    self._assertParseError('[]')