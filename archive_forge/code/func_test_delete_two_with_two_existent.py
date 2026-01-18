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
def test_delete_two_with_two_existent(self):
    self.run_command('delete 1234 5678')
    self.assert_called('DELETE', '/servers/1234', pos=-5)
    self.assert_called('DELETE', '/servers/5678', pos=-1)
    self.run_command('delete sample-server sample-server2')
    self.assert_called('GET', '/servers?name=sample-server', pos=-6)
    self.assert_called('GET', '/servers/1234', pos=-5)
    self.assert_called('DELETE', '/servers/1234', pos=-4)
    self.assert_called('GET', '/servers?name=sample-server2', pos=-3)
    self.assert_called('GET', '/servers/5678', pos=-2)
    self.assert_called('DELETE', '/servers/5678', pos=-1)