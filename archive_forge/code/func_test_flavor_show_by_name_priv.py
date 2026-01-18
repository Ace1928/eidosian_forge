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
def test_flavor_show_by_name_priv(self):
    self.run_command(['flavor-show', '512 MiB Server'])
    self.assert_called('GET', '/flavors/512 MiB Server', pos=0)
    self.assert_called('GET', '/flavors?is_public=None', pos=1)
    self.assert_called('GET', '/flavors/2', pos=2)
    self.assert_called('GET', '/flavors/2/os-extra_specs', pos=3)