import argparse
import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import fixture
import requests_mock
from testtools import matchers
from novaclient import api_versions
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import fake_actions_module
from novaclient.tests.unit import utils
@mock.patch.object(novaclient.shell.OpenStackComputeShell, 'main')
def test_main_error_handling(self, mock_compute_shell):

    class MyException(Exception):
        pass
    with mock.patch('sys.stderr', io.StringIO()):
        mock_compute_shell.side_effect = MyException('message')
        self.assertRaises(SystemExit, novaclient.shell.main, [])
        err = sys.stderr.getvalue()
    self.assertIn('ERROR (MyException): message\n', err)
    self.assertIn('nova CLI is deprecated and will be removed in a future release', err)