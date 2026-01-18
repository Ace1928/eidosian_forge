import socket
from unittest import mock
import uuid
from cinderclient.v3 import client as cinderclient
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import strutils
from glance.common import wsgi
from glance.tests import functional
@mock.patch.object(base, 'connector')
@mock.patch.object(cinderclient, 'Client')
@mock.patch.object(cinder.Store, 'temporary_chown')
@mock.patch.object(cinder, 'connector')
@mock.patch.object(cinder, 'open')
@mock.patch('glance_store._drivers.cinder.Store._wait_volume_status')
@mock.patch.object(strutils, 'mask_dict_password')
@mock.patch.object(socket, 'getaddrinfo')
def test_migrate_image_after_upgrade_not_owner(self, mock_host_addr, mock_mask_pass, mock_wait, mock_open, mock_connector, mock_chown, mocked_cc, mock_base):
    """Test to check if an image is successfully migrated when we upgrade
        from a single cinder store to multiple cinder stores, and that
        GETs from non-owners in the meantime are not interrupted.
        """
    self.setup_single_store()
    self.start_server()
    mocked_cc.return_value = self.cinder_store_mock
    mock_wait.side_effect = self._mock_wait_volume_status
    mock_host_addr.return_value = self.fake_socket_return
    image_id = self._create_and_import(stores=['store1'], extra={'visibility': 'public', 'owner': 'someoneelse'})
    image = self.api_get('/v2/images/%s' % image_id).json
    self.assertEqual('cinder://%s' % self.vol_id, image['locations'][0]['url'])
    self.unset_single_store()
    self.setup_multiple_stores()
    cinder.keystone_sc = mock.MagicMock()
    resp = self.api_get('/v2/images/%s' % image_id, headers={'X-Roles': 'reader'})
    image = resp.json
    self.assertEqual('cinder://store1/%s' % self.vol_id, image['locations'][0]['url'])
    self.assertEqual('store1', image['locations'][0]['metadata']['store'])
    image = self.api_get('/v2/images/%s' % image_id).json
    self.assertEqual('cinder://store1/%s' % self.vol_id, image['locations'][0]['url'])
    mocked_cc.assert_called()
    mock_open.assert_called()
    mock_chown.assert_called()
    mock_connector.get_connector_properties.assert_called()