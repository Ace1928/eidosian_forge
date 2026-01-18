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
@requests_mock.mock()
def test_download_context(self, m):
    """Verify context (ie token) is passed to swift on download."""
    self.config(group='swift1', swift_store_multi_tenant=True)
    store = Store(self.conf, backend='swift1')
    store.configure()
    uri = 'swift+http://127.0.0.1/glance_123/123'
    loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
    m.get('http://127.0.0.1/glance_123/123', headers={'Content-Length': '0'})
    store.get(loc, context=self.ctx)
    self.assertEqual(b'0123', m.last_request.headers['X-Auth-Token'])