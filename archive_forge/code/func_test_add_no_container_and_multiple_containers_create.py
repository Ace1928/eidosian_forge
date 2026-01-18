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
def test_add_no_container_and_multiple_containers_create(self):
    """
        Tests that adding an image with a non-existing container while using
        multi containers will create the container automatically if flag is set
        """
    expected_swift_size = FIVE_KB
    expected_swift_contents = b'*' * expected_swift_size
    expected_checksum = md5(expected_swift_contents, usedforsecurity=False).hexdigest()
    expected_image_id = str(uuid.uuid4())
    container = 'randomname_' + expected_image_id[:2]
    loc = 'swift+config://ref1/%s/%s'
    expected_location = loc % (container, expected_image_id)
    image_swift = io.BytesIO(expected_swift_contents)
    global SWIFT_PUT_OBJECT_CALLS
    SWIFT_PUT_OBJECT_CALLS = 0
    conf = copy.deepcopy(SWIFT_CONF)
    conf['swift_store_user'] = 'tenant:user'
    conf['swift_store_create_container_on_put'] = True
    conf['swift_store_container'] = 'randomname'
    conf['swift_store_multiple_containers_seed'] = 2
    self.config(group='swift1', **conf)
    importlib.reload(swift)
    self.mock_keystone_client()
    self.store = Store(self.conf, backend='swift1')
    self.store.configure()
    loc, size, checksum, metadata = self.store.add(expected_image_id, image_swift, expected_swift_size)
    self.assertEqual('swift1', metadata['store'])
    self.assertEqual(expected_location, loc)
    self.assertEqual(expected_swift_size, size)
    self.assertEqual(expected_checksum, checksum)
    self.assertEqual(1, SWIFT_PUT_OBJECT_CALLS)
    loc = location.get_location_from_uri_and_backend(expected_location, 'swift1', conf=self.conf)
    new_image_swift, new_image_size = self.store.get(loc)
    new_image_contents = b''.join([chunk for chunk in new_image_swift])
    new_image_swift_size = len(new_image_swift)
    self.assertEqual(expected_swift_contents, new_image_contents)
    self.assertEqual(expected_swift_size, new_image_swift_size)