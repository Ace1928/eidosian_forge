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
def test_add_no_container_and_multiple_containers_no_create(self):
    """
        Tests that adding an image with a non-existing container while using
        multiple containers raises an appropriate exception
        """
    conf = copy.deepcopy(SWIFT_CONF)
    conf['swift_store_user'] = 'tenant:user'
    conf['swift_store_create_container_on_put'] = False
    conf['swift_store_container'] = 'randomname'
    conf['swift_store_multiple_containers_seed'] = 2
    self.config(group='swift1', **conf)
    importlib.reload(swift)
    self.mock_keystone_client()
    expected_image_id = str(uuid.uuid4())
    expected_container = 'randomname_' + expected_image_id[:2]
    self.store = Store(self.conf, backend='swift1')
    self.store.configure()
    image_swift = io.BytesIO(b'nevergonnamakeit')
    global SWIFT_PUT_OBJECT_CALLS
    SWIFT_PUT_OBJECT_CALLS = 0
    exception_caught = False
    try:
        self.store.add(expected_image_id, image_swift, 0)
    except exceptions.BackendException as e:
        exception_caught = True
        expected_msg = 'container %s does not exist in Swift'
        expected_msg = expected_msg % expected_container
        self.assertIn(expected_msg, encodeutils.exception_to_unicode(e))
    self.assertTrue(exception_caught)
    self.assertEqual(0, SWIFT_PUT_OBJECT_CALLS)