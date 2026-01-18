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
def test_migration_list_limit_marker_v259(self):
    out, _ = self.run_command('migration-list --limit 10 --marker %s' % FAKE_UUID_1, api_version='2.59')
    self.assert_called('GET', '/os-migrations?limit=10&marker=%s' % FAKE_UUID_1)
    self.assertIn('| UUID', out)