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
@mock.patch.object(cliutils, 'print_list', mock.Mock())
def test_list_with_public_shares(self):
    listed_fields = ['ID', 'Name', 'Size', 'Share Proto', 'Status', 'Is Public', 'Share Type Name', 'Host', 'Availability Zone', 'Project ID']
    self.run_command('list --public')
    self.assert_called('GET', '/shares/detail?is_public=True')
    cliutils.print_list.assert_called_with(mock.ANY, listed_fields, sortby_index=None)