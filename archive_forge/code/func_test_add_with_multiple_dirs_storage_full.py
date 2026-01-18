import builtins
import errno
import hashlib
import io
import json
import os
import stat
from unittest import mock
import uuid
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import filesystem
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_add_with_multiple_dirs_storage_full(self):
    """
        Test StorageFull exception is raised if no filesystem directory
        is found that can store an image.
        """
    store_map = [self.useFixture(fixtures.TempDir()).path, self.useFixture(fixtures.TempDir()).path]
    self.conf.set_override('filesystem_store_datadir', override=None, group='glance_store')
    self.conf.set_override('filesystem_store_datadirs', [store_map[0] + ':100', store_map[1] + ':200'], group='glance_store')
    self.store.configure_add()

    def fake_get_capacity_info(mount_point):
        return 0
    with mock.patch.object(self.store, '_get_capacity_info') as capacity:
        capacity.return_value = 0
        filesystem.ChunkedFile.CHUNKSIZE = units.Ki
        expected_image_id = str(uuid.uuid4())
        expected_file_size = 5 * units.Ki
        expected_file_contents = b'*' * expected_file_size
        image_file = io.BytesIO(expected_file_contents)
        self.assertRaises(exceptions.StorageFull, self.store.add, expected_image_id, image_file, expected_file_size, self.hash_algo)