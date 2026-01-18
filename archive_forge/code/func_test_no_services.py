import unittest
from unittest import mock
from importlib import reload
from os_ken.cmd.manager import main
@mock.patch('sys.argv', new=['osken-manager', '--verbose', 'os_ken.tests.unit.cmd.dummy_app'])
def test_no_services(self):
    self._reset_globals()
    main()
    self._reset_globals()