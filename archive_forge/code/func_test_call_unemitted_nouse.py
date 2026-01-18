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
@mock.patch.object(sys, 'stderr', io.StringIO())
@mock.patch.object(novaclient.shell.DeprecatedAction, '_get_action')
def test_call_unemitted_nouse(self, mock_get_action):
    obj = novaclient.shell.DeprecatedAction('option_strings', 'dest')
    obj('parser', 'namespace', 'values', 'option_string')
    self.assertEqual(obj.emitted, set(['option_string']))
    mock_get_action.assert_called_once_with('parser')
    mock_get_action.return_value.assert_called_once_with('parser', 'namespace', 'values', 'option_string')
    self.assertEqual(sys.stderr.getvalue(), 'WARNING: Option "option_string" is deprecated\n')