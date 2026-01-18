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
def test_aggregate_cache_images(self):
    self.run_command('aggregate-cache-images 1 %s %s' % (FAKE_UUID_1, FAKE_UUID_2), api_version='2.81')
    body = {'cache': [{'id': FAKE_UUID_1}, {'id': FAKE_UUID_2}]}
    self.assert_called('POST', '/os-aggregates/1/images', body)