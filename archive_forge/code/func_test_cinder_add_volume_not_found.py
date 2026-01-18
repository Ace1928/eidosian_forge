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
def test_cinder_add_volume_not_found(self):
    image_file = mock.MagicMock()
    fake_image_id = str(uuid.uuid4())
    expected_size = 0
    fake_volumes = mock.MagicMock(create=mock.MagicMock(side_effect=cinder.cinder_exception.NotFound(code=404)))
    with mock.patch.object(cinder.Store, 'get_cinderclient') as mock_cc:
        mock_cc.return_value = mock.MagicMock(volumes=fake_volumes)
        self.assertRaises(exceptions.BackendException, self.store.add, fake_image_id, image_file, expected_size, self.hash_algo, self.context, None)