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
def test_boot_with_hypervisor_hostname_pre_v274(self):
    cmd = 'boot --flavor 1 --image %s --nic auto --hypervisor-hostname new-host some-server' % FAKE_UUID_1
    self.assertRaises(SystemExit, self.run_command, cmd, api_version='2.73')