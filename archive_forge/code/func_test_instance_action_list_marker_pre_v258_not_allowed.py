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
def test_instance_action_list_marker_pre_v258_not_allowed(self):
    cmd = 'instance-action-list sample-server --marker %s'
    self.assertRaises(SystemExit, self.run_command, cmd % FAKE_UUID_1, api_version='2.57')