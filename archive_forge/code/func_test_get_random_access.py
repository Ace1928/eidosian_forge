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
def test_get_random_access(self):
    """Test a "normal" retrieval of an image in chunks."""
    image_id = str(uuid.uuid4())
    file_contents = b'chunk00000remainder'
    image_file = io.BytesIO(file_contents)
    loc, size, checksum, multihash, _ = self.store.add(image_id, image_file, len(file_contents), self.hash_algo)
    uri = 'file:///%s/%s' % (self.test_dir, image_id)
    loc = location.get_location_from_uri(uri, conf=self.conf)
    data = b''
    for offset in range(len(file_contents)):
        image_file, image_size = self.store.get(loc, offset=offset, chunk_size=1)
        for chunk in image_file:
            data += chunk
    self.assertEqual(file_contents, data)
    data = b''
    chunk_size = 5
    image_file, image_size = self.store.get(loc, offset=chunk_size, chunk_size=chunk_size)
    for chunk in image_file:
        data += chunk
    self.assertEqual(b'00000', data)
    self.assertEqual(chunk_size, image_size)