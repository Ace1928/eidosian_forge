import itertools
from unittest import mock
import ddt
import fixtures
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient import client
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils as test_utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient import utils
from manilaclient.v2 import messages
from manilaclient.v2 import security_services
from manilaclient.v2 import share_access_rules
from manilaclient.v2 import share_group_types
from manilaclient.v2 import share_groups
from manilaclient.v2 import share_instances
from manilaclient.v2 import share_network_subnets
from manilaclient.v2 import share_networks
from manilaclient.v2 import share_servers
from manilaclient.v2 import share_snapshots
from manilaclient.v2 import share_types
from manilaclient.v2 import shares
from manilaclient.v2 import shell as shell_v2
def test_allow_access_with_access_level(self):
    aliases = ['--access_level', '--access-level']
    expected = {'allow_access': {'access_type': 'ip', 'access_to': '10.0.0.6', 'access_level': 'ro'}}
    for alias in aliases:
        for s in self.separators:
            self.run_command('access-allow ' + alias + s + 'ro 1111 ip 10.0.0.6')
            self.assert_called('POST', '/shares/1111/action', body=expected)