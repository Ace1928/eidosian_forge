from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
@ddt.data(type('ShareNetworkUUID', (object,), {'uuid': 'fake_nw'}), type('ShareNetworkID', (object,), {'id': 'fake_nw'}), 'fake_nw')
def test_create_share_with_share_network(self, share_network):
    expected = {'size': 1, 'snapshot_id': None, 'name': None, 'description': None, 'metadata': dict(), 'share_proto': 'nfs', 'share_network_id': 'fake_nw', 'share_type': None, 'is_public': False, 'availability_zone': None, 'scheduler_hints': dict()}
    cs.shares.create('nfs', 1, share_network=share_network)
    cs.assert_called('POST', '/shares', {'share': expected})