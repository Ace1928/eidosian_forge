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
def test_list_fields_redundant(self):
    output, _err = self.run_command('list --fields id,status,status')
    header = output.splitlines()[1]
    self.assertEqual(1, header.count('ID'))
    self.assertEqual(0, header.count('Id'))
    self.assertEqual(1, header.count('Status'))