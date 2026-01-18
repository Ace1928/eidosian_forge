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
def test_migration_list_with_user_id_v280(self):
    user_id = '13cc0930d27c4be0acc14d7c47a3e1f7'
    out = self.run_command('migration-list --user-id %s' % user_id, api_version='2.80')[0]
    self.assert_called('GET', '/os-migrations?user_id=%s' % user_id)
    self.assertIn('User ID', out)
    self.assertIn('Project ID', out)