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
def test_add_with_verifier_small(self):
    """Test that the verifier is updated for smaller images."""
    swift_size = FIVE_KB
    base_byte = b'12345678'
    swift_contents = base_byte * (swift_size // 8)
    image_id = str(uuid.uuid4())
    image_swift = io.BytesIO(swift_contents)
    self.store = Store(self.conf, backend='swift1')
    self.store.configure()
    orig_max_size = self.store.large_object_size
    orig_temp_size = self.store.large_object_chunk_size
    custom_size = 6 * units.Ki
    verifier = mock.MagicMock(name='mock_verifier')
    try:
        self.store.large_object_size = custom_size
        self.store.large_object_chunk_size = custom_size
        self.store.add(image_id, image_swift, swift_size, verifier=verifier)
    finally:
        self.store.large_object_chunk_size = orig_temp_size
        self.store.large_object_size = orig_max_size
    self.assertEqual(2, verifier.update.call_count)
    swift_contents_piece = base_byte * (swift_size // 8)
    calls = [mock.call(swift_contents_piece), mock.call(b'')]
    verifier.update.assert_has_calls(calls)