from unittest import mock
import uuid
from cinderclient import exceptions as cinder_exc
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients.os import cinder
from heat.tests import common
from heat.tests import utils
def test_get_snapshot(self):
    """Tests the get_volume_snapshot function."""
    snapshot_id = str(uuid.uuid4())
    my_snapshot = mock.MagicMock()
    self.cinder_client.volume_snapshots.get.return_value = my_snapshot
    self.assertEqual(my_snapshot, self.cinder_plugin.get_volume_snapshot(snapshot_id))
    self.cinder_client.volume_snapshots.get.assert_called_once_with(snapshot_id)