from unittest import mock
from oslo_config import cfg
from oslotest import base
from cinderclient import exceptions as cinder_exception
from glance_store.common import attachment_state_manager as attach_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
@mock.patch.object(cinder_utils.API, 'attachment_delete')
def test_detach_without_attach(self, mock_attach_delete):
    ex = exceptions.BackendException
    conn = mock.MagicMock()
    mock_attach_delete.side_effect = ex()
    self.assertRaises(ex, self._sentinel_detach, conn)
    conn.disconnect_volume.assert_called_once_with(*self.disconnect_vol_call)