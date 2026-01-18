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
def test_create_server_group_with_invalid_value(self):
    result = self.assertRaises(exceptions.CommandError, self.run_command, 'server-group-create sg1 anti-affinity --rule max_server_per_host=foo', api_version='2.64')
    self.assertIn("Invalid 'max_server_per_host' value: foo", str(result))