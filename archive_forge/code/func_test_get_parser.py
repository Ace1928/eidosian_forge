from unittest import mock
import testtools
from blazarclient import command
from blazarclient import tests
def test_get_parser(self):
    self.command.get_parser('TestCase')
    self.parser.assert_called_once_with('TestCase')