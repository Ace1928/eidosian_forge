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
from glance_store._drivers.swift import buffered
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import backend
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_add_raises_storage_full(self):
    conf = copy.deepcopy(SWIFT_CONF)
    conf['default_swift_reference'] = 'store_2'
    self.config(**conf)
    importlib.reload(swift)
    self.mock_keystone_client()
    self.store = Store(self.conf)
    self.store.configure()

    def fake_put_object_entity_too_large(*args, **kwargs):
        msg = 'Test Out of Quota'
        raise swiftclient.ClientException(msg, http_status=http.client.REQUEST_ENTITY_TOO_LARGE)
    self.useFixture(fixtures.MockPatch('swiftclient.client.put_object', fake_put_object_entity_too_large))
    expected_swift_size = FIVE_KB
    expected_swift_contents = b'*' * expected_swift_size
    expected_image_id = str(uuid.uuid4())
    image_swift = io.BytesIO(expected_swift_contents)
    self.assertRaises(exceptions.StorageFull, self.store.add, expected_image_id, image_swift, expected_swift_size, HASH_ALGO)