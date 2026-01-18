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
def test_boot_with_multiple_trusted_image_certificates(self):
    self.run_command('boot --flavor 1 --image %s --nic auto some-server --trusted-image-certificate-id id1 --trusted-image-certificate-id id2' % FAKE_UUID_1, api_version='2.63')
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'some-server', 'imageRef': FAKE_UUID_1, 'min_count': 1, 'max_count': 1, 'networks': 'auto', 'trusted_image_certificates': ['id1', 'id2']}})