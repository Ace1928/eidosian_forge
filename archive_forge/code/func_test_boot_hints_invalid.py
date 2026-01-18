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
def test_boot_hints_invalid(self):
    cmd = 'boot --image %s --flavor 1 --hint a0cf03a5-d921-4877-bb5c-86d26cf818e1 some-server' % FAKE_UUID_1
    _, err = self.run_command(cmd, expected_error=SystemExit)
    self.assertIn("'a0cf03a5-d921-4877-bb5c-86d26cf818e1' is not in the format of 'key=value'", err)