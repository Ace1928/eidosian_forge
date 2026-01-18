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
def test_boot_files(self):
    testfile = os.path.join(os.path.dirname(__file__), 'testfile.txt')
    with open(testfile) as testfile_fd:
        data = testfile_fd.read()
    expected = base64.b64encode(data.encode('utf-8')).decode('utf-8')
    cmd = 'boot some-server --flavor 1 --image %s --file /tmp/foo=%s --file /tmp/bar=%s'
    self.run_command(cmd % (FAKE_UUID_1, testfile, testfile))
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'some-server', 'imageRef': FAKE_UUID_1, 'min_count': 1, 'max_count': 1, 'personality': [{'path': '/tmp/bar', 'contents': expected}, {'path': '/tmp/foo', 'contents': expected}]}})