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
@ddt.data('--group sg1313', '--share-group sg1313', '--share_group sg1313')
@mock.patch.object(shell_v2, '_find_share_group', mock.Mock())
def test_delete_with_share_group(self, sg_cmd):
    fake_sg = type('FakeShareGroup', (object,), {'id': sg_cmd.split()[-1]})
    shell_v2._find_share_group.return_value = fake_sg
    self.run_command('delete 1234 %s' % sg_cmd)
    self.assert_called('DELETE', '/shares/1234?share_group_id=sg1313')
    self.assertTrue(shell_v2._find_share_group.called)