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
def test_boot_nics_auto_allocate_default(self):
    """Tests that if microversion>=2.37 is specified and no --nics are
        specified that a single --nic with net-id=auto is used.
        """
    cmd = 'boot --image %s --flavor 1 some-server' % FAKE_UUID_1
    self.run_command(cmd, api_version='2.37')
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'some-server', 'imageRef': FAKE_UUID_1, 'min_count': 1, 'max_count': 1, 'networks': 'auto'}})