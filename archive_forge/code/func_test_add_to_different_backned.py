import builtins
import errno
import io
import json
import os
import stat
from unittest import mock
import uuid
import fixtures
from oslo_config import cfg
from oslo_utils.secretutils import md5
from oslo_utils import units
import glance_store as store
from glance_store._drivers import filesystem
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_add_to_different_backned(self):
    """Test that we can add an image via the filesystem backend."""
    self.store = filesystem.Store(self.conf, backend='file2')
    self.config(filesystem_store_datadir=self.test_dir, group='file2')
    self.store.configure()
    self.register_store_backend_schemes(self.store, 'file', 'file2')
    filesystem.ChunkedFile.CHUNKSIZE = units.Ki
    expected_image_id = str(uuid.uuid4())
    expected_file_size = 5 * units.Ki
    expected_file_contents = b'*' * expected_file_size
    expected_checksum = md5(expected_file_contents, usedforsecurity=False).hexdigest()
    expected_location = 'file://%s/%s' % (self.test_dir, expected_image_id)
    image_file = io.BytesIO(expected_file_contents)
    loc, size, checksum, metadata = self.store.add(expected_image_id, image_file, expected_file_size)
    self.assertEqual(expected_location, loc)
    self.assertEqual(expected_file_size, size)
    self.assertEqual(expected_checksum, checksum)
    self.assertEqual('file2', metadata['store'])
    uri = 'file:///%s/%s' % (self.test_dir, expected_image_id)
    loc = location.get_location_from_uri_and_backend(uri, 'file2', conf=self.conf)
    new_image_file, new_image_size = self.store.get(loc)
    new_image_contents = b''
    new_image_file_size = 0
    for chunk in new_image_file:
        new_image_file_size += len(chunk)
        new_image_contents += chunk
    self.assertEqual(expected_file_contents, new_image_contents)
    self.assertEqual(expected_file_size, new_image_file_size)