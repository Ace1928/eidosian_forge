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
def test_meta_parsing(self):
    meta = ['key1=meta1', 'key2=meta2']
    ref = {'key1': 'meta1', 'key2': 'meta2'}
    parsed_meta = novaclient.v2.shell._meta_parsing(meta)
    self.assertEqual(ref, parsed_meta)