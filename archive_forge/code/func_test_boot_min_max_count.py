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
def test_boot_min_max_count(self):
    self.run_command('boot --image %s --flavor 1 --max-count 3 server' % FAKE_UUID_1)
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'server', 'imageRef': FAKE_UUID_1, 'min_count': 1, 'max_count': 3}})
    self.run_command('boot --image %s --flavor 1 --min-count 3 server' % FAKE_UUID_1)
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'server', 'imageRef': FAKE_UUID_1, 'min_count': 3, 'max_count': 3}})
    self.run_command('boot --image %s --flavor 1 --min-count 3 --max-count 3 server' % FAKE_UUID_1)
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'server', 'imageRef': FAKE_UUID_1, 'min_count': 3, 'max_count': 3}})
    self.run_command('boot --image %s --flavor 1 --min-count 3 --max-count 5 server' % FAKE_UUID_1)
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'server', 'imageRef': FAKE_UUID_1, 'min_count': 3, 'max_count': 5}})
    cmd = 'boot --image %s --flavor 1 --min-count 3 --max-count 1 serv' % FAKE_UUID_1
    self.assertRaises(exceptions.CommandError, self.run_command, cmd)