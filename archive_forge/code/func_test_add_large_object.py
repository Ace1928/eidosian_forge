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
@mock.patch('glance_store._drivers.swift.utils.is_multiple_swift_store_accounts_enabled', mock.Mock(return_value=True))
def test_add_large_object(self):
    """
        Tests that adding a very large image. We simulate the large
        object by setting store.large_object_size to a small number
        and then verify that there have been a number of calls to
        put_object()...
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
    try:
        self.store.large_object_size = units.Ki
        self.store.large_object_chunk_size = units.Ki
        loc, size, checksum, metadata = self.store.add(expected_image_id, image_swift, expected_swift_size)
    finally:
        self.store.large_object_chunk_size = orig_temp_size
        self.store.large_object_size = orig_max_size
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