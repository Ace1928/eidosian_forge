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
def test_aggregate_add_host_by_id_v2_41(self):
    out, err = self.run_command('aggregate-add-host 1 host1', api_version='2.41')
    body = {'add_host': {'host': 'host1'}}
    self.assert_called('POST', '/os-aggregates/1/action', body, pos=-2)
    self.assert_called('GET', '/os-aggregates/1', pos=-1)
    self.assertIn('UUID', out)
    self.assertIn('80785864-087b-45a5-a433-b20eac9b58aa', out)