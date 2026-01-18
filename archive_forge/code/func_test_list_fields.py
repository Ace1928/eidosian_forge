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
def test_list_fields(self):
    output, _err = self.run_command('list --fields host,security_groups,OS-EXT-MOD:some_thing')
    self.assert_called('GET', '/servers/detail')
    self.assertIn('computenode1', output)
    self.assertIn('securitygroup1', output)
    self.assertIn('OS-EXT-MOD: Some Thing', output)
    self.assertIn('mod_some_thing_value', output)
    output, _err = self.run_command('list --fields networks')
    self.assertIn('Networks', output)
    self.assertIn('10.11.12.13', output)
    self.assertIn('5.6.7.8', output)