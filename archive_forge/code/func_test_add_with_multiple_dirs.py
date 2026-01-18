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
def test_add_with_multiple_dirs(self):
    """Test adding multiple filesystem directories."""
    store_map = [self.useFixture(fixtures.TempDir()).path, self.useFixture(fixtures.TempDir()).path]
    self.conf.set_override('filesystem_store_datadir', override=None, group='glance_store')
    self.conf.set_override('filesystem_store_datadirs', [store_map[0] + ':100', store_map[1] + ':200'], group='glance_store')
    self.store.configure()
    filesystem.ChunkedFile.CHUNKSIZE = units.Ki
    expected_image_id = str(uuid.uuid4())
    expected_file_size = 5 * units.Ki
    expected_file_contents = b'*' * expected_file_size
    expected_checksum = md5(expected_file_contents, usedforsecurity=False).hexdigest()
    expected_multihash = hashlib.sha256(expected_file_contents).hexdigest()
    expected_location = 'file://%s/%s' % (store_map[1], expected_image_id)
    image_file = io.BytesIO(expected_file_contents)
    loc, size, checksum, multihash, _ = self.store.add(expected_image_id, image_file, expected_file_size, self.hash_algo)
    self.assertEqual(expected_location, loc)
    self.assertEqual(expected_file_size, size)
    self.assertEqual(expected_checksum, checksum)
    self.assertEqual(expected_multihash, multihash)
    loc = location.get_location_from_uri(expected_location, conf=self.conf)
    new_image_file, new_image_size = self.store.get(loc)
    new_image_contents = b''
    new_image_file_size = 0
    for chunk in new_image_file:
        new_image_file_size += len(chunk)
        new_image_contents += chunk
    self.assertEqual(expected_file_contents, new_image_contents)
    self.assertEqual(expected_file_size, new_image_file_size)