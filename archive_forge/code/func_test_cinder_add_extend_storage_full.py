import contextlib
import hashlib
import io
import math
import os
from unittest import mock
import socket
import sys
import tempfile
import time
import uuid
from keystoneauth1 import exceptions as keystone_exc
from os_brick.initiator import connector
from oslo_concurrency import processutils
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers.cinder import scaleio
from glance_store.common import attachment_state_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store import location
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
def test_cinder_add_extend_storage_full(self):
    expected_volume_size = 2 * units.Gi
    fakebuffer = mock.MagicMock()
    fakebuffer.__len__.return_value = int(expected_volume_size / 2)
    expected_image_id = str(uuid.uuid4())
    expected_volume_id = str(uuid.uuid4())
    expected_size = 0
    image_file = mock.MagicMock(read=mock.MagicMock(side_effect=[fakebuffer, fakebuffer, None]))
    fake_volume = mock.MagicMock(id=expected_volume_id, status='available', size=1)
    verifier = None
    fake_client = mock.MagicMock()
    fake_volume.manager.get.return_value = fake_volume
    fake_volumes = mock.MagicMock(create=mock.Mock(return_value=fake_volume))
    with mock.patch.object(cinder.Store, 'get_cinderclient') as mock_cc, mock.patch.object(self.store, '_open_cinder_volume'), mock.patch.object(cinder.utils, 'get_hasher'), mock.patch.object(cinder.Store, '_wait_volume_status') as mock_wait:
        mock_cc.return_value = mock.MagicMock(client=fake_client, volumes=fake_volumes)
        mock_wait.side_effect = [fake_volume, exceptions.BackendException]
        self.assertRaises(exceptions.StorageFull, self.store.add, expected_image_id, image_file, expected_size, self.hash_algo, self.context, verifier)