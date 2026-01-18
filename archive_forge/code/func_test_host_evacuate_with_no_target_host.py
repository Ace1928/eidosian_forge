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
def test_host_evacuate_with_no_target_host(self):
    self.run_command('host-evacuate --on-shared-storage hyper')
    self.assert_called('GET', '/os-hypervisors/hyper/servers', pos=0)
    self.assert_called('POST', '/servers/uuid1/action', {'evacuate': {'onSharedStorage': True}}, pos=1)
    self.assert_called('POST', '/servers/uuid2/action', {'evacuate': {'onSharedStorage': True}}, pos=2)
    self.assert_called('POST', '/servers/uuid3/action', {'evacuate': {'onSharedStorage': True}}, pos=3)
    self.assert_called('POST', '/servers/uuid4/action', {'evacuate': {'onSharedStorage': True}}, pos=4)