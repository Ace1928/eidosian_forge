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
def test_connection_manager_does_not_use_endpoint_from_catalog(self):
    self.store.configure()
    self.context.service_catalog = [{'endpoint_links': [], 'endpoints': [{'region': 'RegionOne', 'publicURL': 'https://scexample.com'}], 'type': 'object-store', 'name': 'Object Storage Service'}]
    connection_manager = manager.MultiTenantConnectionManager(store=self.store, store_location=self.location, context=self.context)
    conn = connection_manager._init_connection()
    self.assertNotEqual('https://scexample.com', conn.preauthurl)
    self.assertEqual('https://example.com', conn.preauthurl)