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
def test_keypair_list_with_user_id_limit_and_marker(self):
    self.run_command('keypair-list --user test_user --marker test_kp --limit 3', api_version='2.35')
    self.assert_called('GET', '/os-keypairs?limit=3&marker=test_kp&user_id=test_user')