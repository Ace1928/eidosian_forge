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
def test_boot_invalid_files(self):
    invalid_file = os.path.join(os.path.dirname(__file__), 'asdfasdfasdfasdf')
    cmd = 'boot some-server --flavor 1 --image %s --file /foo=%s' % (FAKE_UUID_1, invalid_file)
    self.assertRaises(exceptions.CommandError, self.run_command, cmd)