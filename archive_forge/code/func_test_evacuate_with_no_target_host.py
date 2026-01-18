import argparse
import base64
import builtins
import collections
import datetime
import io
import os
from unittest import mock
import fixtures
from oslo_utils import timeutils
import testtools
import novaclient
from novaclient import api_versions
from novaclient import base
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
import novaclient.v2.shell
def test_evacuate_with_no_target_host(self):
    self.run_command('evacuate sample-server')
    self.assert_called('POST', '/servers/1234/action', {'evacuate': {'onSharedStorage': False}})
    self.run_command('evacuate sample-server --password NewAdminPass')
    self.assert_called('POST', '/servers/1234/action', {'evacuate': {'onSharedStorage': False, 'adminPass': 'NewAdminPass'}})
    self.run_command('evacuate sample-server --on-shared-storage')
    self.assert_called('POST', '/servers/1234/action', {'evacuate': {'onSharedStorage': True}})