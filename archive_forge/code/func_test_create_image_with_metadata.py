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
def test_create_image_with_metadata(self):
    self.run_command('image-create sample-server mysnapshot --metadata mykey=123')
    self.assert_called('POST', '/servers/1234/action', {'createImage': {'name': 'mysnapshot', 'metadata': {'mykey': '123'}}})