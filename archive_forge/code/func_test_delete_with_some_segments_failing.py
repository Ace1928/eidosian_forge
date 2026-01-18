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
def test_delete_with_some_segments_failing(self):
    """
        Tests that delete of a segmented object recovers from error(s) while
        deleting one or more segments.
        To test this we add a segmented object first and then delete it, while
        simulating errors on one or more segments.
        """
    test_image_id = str(uuid.uuid4())

    def fake_head_object(container, object_name):
        object_manifest = '/'.join([container, object_name]) + '-'
        return {'x-object-manifest': object_manifest}

    def fake_get_container(container, **kwargs):
        return (None, [{'name': '%s-%03d' % (test_image_id, x)} for x in range(1, 6)])

    def fake_delete_object(container, object_name):
        global SWIFT_DELETE_OBJECT_CALLS
        SWIFT_DELETE_OBJECT_CALLS += 1
        if object_name.endswith('-001') or object_name.endswith('-003'):
            raise swiftclient.ClientException('Object DELETE failed')
        else:
            pass
    conf = copy.deepcopy(SWIFT_CONF)
    self.config(group='swift1', **conf)
    importlib.reload(swift)
    self.store = Store(self.conf, backend='swift1')
    self.store.configure()
    loc_uri = 'swift+https://%s:key@localhost:8080/glance/%s'
    loc_uri = loc_uri % (self.swift_store_user, test_image_id)
    loc = location.get_location_from_uri_and_backend(loc_uri, 'swift1', conf=self.conf)
    conn = self.store.get_connection(loc.store_location)
    conn.delete_object = fake_delete_object
    conn.head_object = fake_head_object
    conn.get_container = fake_get_container
    global SWIFT_DELETE_OBJECT_CALLS
    SWIFT_DELETE_OBJECT_CALLS = 0
    self.store.delete(loc, connection=conn)
    self.assertEqual(6, SWIFT_DELETE_OBJECT_CALLS)