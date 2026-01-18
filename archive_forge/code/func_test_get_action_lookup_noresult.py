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
def test_get_action_lookup_noresult(self):
    parser = mock.Mock(**{'_registry_get.return_value': None})
    obj = novaclient.shell.DeprecatedAction('option_strings', 'dest', real_action='store', const=1)
    result = obj._get_action(parser)
    self.assertIsNone(result)
    self.assertIsNone(obj.real_action)
    parser._registry_get.assert_called_once_with('action', 'store')
    self.assertEqual(sys.stderr.getvalue(), 'WARNING: Programming error: Unknown real action "store"\n')