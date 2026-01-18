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
def test_boot_nics_net_name_neutron_blank(self):
    cmd = 'boot --image %s --flavor 1 --nic net-name=blank some-server' % FAKE_UUID_1
    msg = 'No Network matching blank\\..*'
    with testtools.ExpectedException(exceptions.CommandError, msg):
        self.run_command(cmd)