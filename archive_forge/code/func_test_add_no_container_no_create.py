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
def test_add_no_container_no_create(self):
    """
        Tests that adding an image with a non-existing container
        raises an appropriate exception
        """
    conf = copy.deepcopy(SWIFT_CONF)
    conf['swift_store_user'] = 'tenant:user'
    conf['swift_store_create_container_on_put'] = False
    conf['swift_store_container'] = 'noexist'
    self.config(group='swift1', **conf)
    importlib.reload(swift)
    self.mock_keystone_client()
    self.store = Store(self.conf, backend='swift1')
    self.store.configure()
    image_swift = io.BytesIO(b'nevergonnamakeit')
    global SWIFT_PUT_OBJECT_CALLS
    SWIFT_PUT_OBJECT_CALLS = 0
    exception_caught = False
    try:
        self.store.add(str(uuid.uuid4()), image_swift, 0)
    except exceptions.BackendException as e:
        exception_caught = True
        self.assertIn('container noexist does not exist in Swift', encodeutils.exception_to_unicode(e))
    self.assertTrue(exception_caught)
    self.assertEqual(0, SWIFT_PUT_OBJECT_CALLS)