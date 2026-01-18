import copy
from unittest import mock
import fixtures
import hashlib
import http.client
import importlib
import io
import tempfile
import uuid
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests_mock
import swiftclient
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
import glance_store.multi_backend as store
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_add_large_object_zero_size(self):
    """
        Tests that adding an image to Swift which has both an unknown size and
        exceeds Swift's maximum limit of 5GB is correctly uploaded.

        We avoid the overhead of creating a 5GB object for this test by
        temporarily setting MAX_SWIFT_OBJECT_SIZE to 1KB, and then adding
        an object of 5KB.

        Bug lp:891738
        """
    expected_swift_size = FIVE_KB
    expected_swift_contents = b'*' * expected_swift_size
    expected_checksum = md5(expected_swift_contents, usedforsecurity=False).hexdigest()
    expected_image_id = str(uuid.uuid4())
    loc = 'swift+config://ref1/glance/%s'
    expected_location = loc % expected_image_id
    image_swift = io.BytesIO(expected_swift_contents)
    global SWIFT_PUT_OBJECT_CALLS
    SWIFT_PUT_OBJECT_CALLS = 0
    self.store = Store(self.conf, backend='swift1')
    self.store.configure()
    orig_max_size = self.store.large_object_size
    orig_temp_size = self.store.large_object_chunk_size
    global MAX_SWIFT_OBJECT_SIZE
    orig_max_swift_object_size = MAX_SWIFT_OBJECT_SIZE
    try:
        MAX_SWIFT_OBJECT_SIZE = units.Ki
        self.store.large_object_size = units.Ki
        self.store.large_object_chunk_size = units.Ki
        loc, size, checksum, metadata = self.store.add(expected_image_id, image_swift, 0)
    finally:
        self.store.large_object_chunk_size = orig_temp_size
        self.store.large_object_size = orig_max_size
        MAX_SWIFT_OBJECT_SIZE = orig_max_swift_object_size
    self.assertEqual('swift1', metadata['store'])
    self.assertEqual(expected_location, loc)
    self.assertEqual(expected_swift_size, size)
    self.assertEqual(expected_checksum, checksum)
    self.assertEqual(6, SWIFT_PUT_OBJECT_CALLS)
    loc = location.get_location_from_uri_and_backend(expected_location, 'swift1', conf=self.conf)
    new_image_swift, new_image_size = self.store.get(loc)
    new_image_contents = b''.join([chunk for chunk in new_image_swift])
    new_image_swift_size = len(new_image_contents)
    self.assertEqual(expected_swift_contents, new_image_contents)
    self.assertEqual(expected_swift_size, new_image_swift_size)