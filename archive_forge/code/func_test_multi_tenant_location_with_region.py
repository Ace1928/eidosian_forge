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
def test_multi_tenant_location_with_region(self):
    self.config(group='swift1', swift_store_region='WestCarolina')
    store = swift.MultiTenantStore(self.conf, backend='swift1')
    store.configure()
    self.ctxt.service_catalog[0]['endpoints'][0]['region'] = 'WestCarolina'
    self.assertEqual('https://some_endpoint', store._get_endpoint(self.ctxt))