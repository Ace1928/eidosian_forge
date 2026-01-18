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
def test_add_with_file_perm_for_owner_users_access(self):
    """
        Test that we can add an image via the filesystem backend with a
        required image file permission.
        """
    store = self.useFixture(fixtures.TempDir()).path
    self.conf.set_override('filesystem_store_datadir', store, group='glance_store')
    self.conf.set_override('filesystem_store_file_perm', 600, group='glance_store')
    os.chmod(store, 448)
    self.assertEqual(448, stat.S_IMODE(os.stat(store)[stat.ST_MODE]))
    self.store.configure_add()
    filesystem.Store.WRITE_CHUNKSIZE = units.Ki
    expected_image_id = str(uuid.uuid4())
    expected_file_size = 5 * units.Ki
    expected_file_contents = b'*' * expected_file_size
    expected_checksum = md5(expected_file_contents, usedforsecurity=False).hexdigest()
    expected_multihash = hashlib.sha256(expected_file_contents).hexdigest()
    expected_location = 'file://%s/%s' % (store, expected_image_id)
    image_file = io.BytesIO(expected_file_contents)
    location, size, checksum, multihash, _ = self.store.add(expected_image_id, image_file, expected_file_size, self.hash_algo)
    self.assertEqual(expected_location, location)
    self.assertEqual(expected_file_size, size)
    self.assertEqual(expected_checksum, checksum)
    self.assertEqual(expected_multihash, multihash)
    self.assertEqual(448, stat.S_IMODE(os.stat(store)[stat.ST_MODE]))
    mode = os.stat(expected_location[len('file:/'):])[stat.ST_MODE]
    perm = int(str(self.conf.glance_store.filesystem_store_file_perm), 8)
    self.assertEqual(perm, stat.S_IMODE(mode))