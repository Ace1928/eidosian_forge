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
def test_host_evacuate_live_with_block_migration_strict(self):
    self.run_command('host-evacuate-live --block-migrate hyper2 --strict')
    self.assert_called('GET', '/os-hypervisors/hyper2/servers', pos=0)
    body = {'os-migrateLive': {'host': None, 'block_migration': True, 'disk_over_commit': False}}
    self.assert_called('POST', '/servers/uuid3/action', body, pos=1)
    self.assert_called('POST', '/servers/uuid4/action', body, pos=2)