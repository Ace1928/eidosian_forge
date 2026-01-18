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
@mock.patch('glance_store._drivers.swift.connection_manager.SingleTenantConnectionManager')
def test_get_connection_manager_single_tenant(self, manager_class):
    manager = mock.MagicMock()
    manager_class.return_value = manager
    store = Store(self.conf, backend='swift1')
    store.configure()
    loc = mock.MagicMock()
    self.assertEqual(store.get_manager(loc), manager)