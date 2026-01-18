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
def test_delete_host_meta(self):
    self.run_command('host-meta hyper delete key1')
    self.assert_called('GET', '/os-hypervisors/hyper/servers', pos=0)
    self.assert_called('DELETE', '/servers/uuid1/metadata/key1', pos=1)
    self.assert_called('DELETE', '/servers/uuid2/metadata/key1', pos=2)