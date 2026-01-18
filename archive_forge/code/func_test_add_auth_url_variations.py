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
def test_add_auth_url_variations(self):
    """
        Test that we can add an image via the swift backend with
        a variety of different auth_address values
        """
    conf = copy.deepcopy(SWIFT_CONF)
    self.config(group='swift1', **conf)
    variations = {'store_4': 'swift+config://store_4/glance/%s', 'store_5': 'swift+config://store_5/glance/%s', 'store_6': 'swift+config://store_6/glance/%s'}
    for variation, expected_location in variations.items():
        image_id = str(uuid.uuid4())
        expected_location = expected_location % image_id
        expected_swift_size = FIVE_KB
        expected_swift_contents = b'*' * expected_swift_size
        expected_checksum = md5(expected_swift_contents, usedforsecurity=False).hexdigest()
        image_swift = io.BytesIO(expected_swift_contents)
        global SWIFT_PUT_OBJECT_CALLS
        SWIFT_PUT_OBJECT_CALLS = 0
        conf['default_swift_reference'] = variation
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        self.mock_keystone_client()
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        loc, size, checksum, metadata = self.store.add(image_id, image_swift, expected_swift_size)
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