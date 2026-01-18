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
def test_list_v2_26_tags_any(self):
    self.run_command('list --tags-any tag1,tag2', api_version='2.26')
    self.assert_called('GET', '/servers/detail?tags-any=tag1%2Ctag2')