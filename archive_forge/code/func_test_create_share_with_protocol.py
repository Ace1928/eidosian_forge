from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
@ddt.data('nfs', 'cifs', 'cephfs', 'glusterfs', 'hdfs', 'maprfs')
def test_create_share_with_protocol(self, protocol):
    expected = {'size': 1, 'snapshot_id': None, 'name': None, 'description': None, 'metadata': dict(), 'share_proto': protocol, 'share_network_id': None, 'share_type': None, 'is_public': False, 'availability_zone': None, 'scheduler_hints': dict()}
    cs.shares.create(protocol, 1)
    cs.assert_called('POST', '/shares', {'share': expected})