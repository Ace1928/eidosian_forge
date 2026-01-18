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
def test_boot_with_trusted_image_certificates_pre_v263(self):
    cmd = 'boot --flavor 1 --image %s some-server --trusted-image-certificate-id id1 --trusted-image-certificate-id id2' % FAKE_UUID_1
    self.assertRaises(SystemExit, self.run_command, cmd, api_version='2.62')