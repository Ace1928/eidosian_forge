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
def test_boot_from_volume_with_volume_type_old_microversion(self):
    ex = self.assertRaises(exceptions.CommandError, self.run_command, 'boot --flavor 1 --block-device id=%s,source=image,dest=volume,size=1,bootindex=0,shutdown=remove,tag=foo,volume_type=lvm bfv-server' % FAKE_UUID_1, api_version='2.66')
    self.assertIn("'volume_type' in block device mapping is not supported in API version", str(ex))