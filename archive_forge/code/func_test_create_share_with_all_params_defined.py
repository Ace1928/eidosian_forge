from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
@ddt.data({'is_public': True, 'availability_zone': 'nova'}, {'is_public': False, 'availability_zone': 'fake_azzzzz'})
@ddt.unpack
def test_create_share_with_all_params_defined(self, is_public, availability_zone):
    body = {'share': {'is_public': is_public, 'share_type': None, 'name': None, 'snapshot_id': None, 'description': None, 'metadata': {}, 'share_proto': 'nfs', 'share_network_id': None, 'size': 1, 'availability_zone': availability_zone, 'scheduler_hints': {}}}
    cs.shares.create('nfs', 1, is_public=is_public, availability_zone=availability_zone)
    cs.assert_called('POST', '/shares', body)