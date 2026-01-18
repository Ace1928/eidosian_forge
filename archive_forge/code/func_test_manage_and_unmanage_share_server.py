import ast
import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@testtools.skipUnless(CONF.run_manage_tests, 'Share Manage/Unmanage tests are disabled.')
@utils.skip_if_microversion_not_supported('2.49')
def test_manage_and_unmanage_share_server(self):
    share, share_network = self._create_share_and_share_network()
    share_server_id = self.client.get_share(self.share['id'])['share_server_id']
    server = self.client.get_share_server(share_server_id)
    server_host = server['host']
    export_location = self.client.list_share_export_locations(self.share['id'])[0]['Path']
    share_host = share['host']
    identifier = server['identifier']
    self.assertEqual('True', server['is_auto_deletable'])
    self.client.unmanage_share(share['id'])
    self.client.wait_for_share_deletion(share['id'])
    server = self.client.get_share_server(share_server_id)
    self.assertEqual('False', server['is_auto_deletable'])
    self.client.unmanage_server(share_server_id)
    self.client.wait_for_share_server_deletion(share_server_id)
    managed_share_server_id = self.client.share_server_manage(server_host, share_network['id'], identifier)
    self.client.wait_for_resource_status(managed_share_server_id, constants.STATUS_ACTIVE, resource_type='share_server')
    managed_server = self.client.get_share_server(managed_share_server_id)
    self.assertEqual('False', managed_server['is_auto_deletable'])
    managed_share_id = self.client.manage_share(share_host, self.protocol, export_location, managed_share_server_id)
    self.client.wait_for_resource_status(managed_share_id, constants.STATUS_AVAILABLE)
    self._delete_share_and_share_server(managed_share_id, managed_share_server_id)
    self.client.delete_share_network(share_network['id'])