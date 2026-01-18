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
def test_get_action_nolookup(self):
    action_class = mock.Mock()
    parser = mock.Mock(**{'_registry_get.return_value': action_class})
    obj = novaclient.shell.DeprecatedAction('option_strings', 'dest', real_action='nothing', const=1)
    obj.real_action = 'action'
    result = obj._get_action(parser)
    self.assertEqual(result, 'action')
    self.assertEqual(obj.real_action, 'action')
    self.assertFalse(parser._registry_get.called)
    self.assertFalse(action_class.called)
    self.assertEqual(sys.stderr.getvalue(), '')