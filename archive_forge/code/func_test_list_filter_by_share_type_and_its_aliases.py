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
def test_list_filter_by_share_type_and_its_aliases(self):
    fake_st = type('Empty', (object,), {'id': 'fake_st'})
    aliases = ['--share-type', '--share_type', '--share-type-id', '--share-type_id', '--share_type-id', '--share_type_id']
    for alias in aliases:
        for separator in self.separators:
            with mock.patch.object(apiclient_utils, 'find_resource', mock.Mock(return_value=fake_st)):
                self.run_command('list ' + alias + separator + fake_st.id)
                self.assert_called('GET', '/shares/detail?share_type_id=' + fake_st.id)