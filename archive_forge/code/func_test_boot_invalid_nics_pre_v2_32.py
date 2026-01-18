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
def test_boot_invalid_nics_pre_v2_32(self):
    cmd = 'boot --image %s --flavor 1 --nic net-id=1,port-id=2 some-server' % FAKE_UUID_1
    ex = self.assertRaises(exceptions.CommandError, self.run_command, cmd, api_version='2.1')
    self.assertNotIn('tag=tag', str(ex))