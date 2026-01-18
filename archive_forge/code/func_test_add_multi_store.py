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
def test_add_multi_store(self):
    conf = copy.deepcopy(SWIFT_CONF)
    conf['default_swift_reference'] = 'store_2'
    self.config(group='swift1', **conf)
    importlib.reload(swift)
    self.mock_keystone_client()
    self.store = Store(self.conf, backend='swift1')
    self.store.configure()
    expected_swift_size = FIVE_KB
    expected_swift_contents = b'*' * expected_swift_size
    expected_image_id = str(uuid.uuid4())
    image_swift = io.BytesIO(expected_swift_contents)
    global SWIFT_PUT_OBJECT_CALLS
    SWIFT_PUT_OBJECT_CALLS = 0
    loc = 'swift+config://store_2/glance/%s'
    expected_location = loc % expected_image_id
    location, size, checksum, arg = self.store.add(expected_image_id, image_swift, expected_swift_size)
    self.assertEqual('swift1', arg['store'])
    self.assertEqual(expected_location, location)