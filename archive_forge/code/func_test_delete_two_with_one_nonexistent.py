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
def test_delete_two_with_one_nonexistent(self):
    cmd = 'delete 1234 123456789'
    self.assertRaises(exceptions.CommandError, self.run_command, cmd)
    self.assert_called_anytime('DELETE', '/servers/1234')
    cmd = 'delete sample-server nonexistentserver'
    self.assertRaises(exceptions.CommandError, self.run_command, cmd)
    self.assert_called_anytime('DELETE', '/servers/1234')