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
def test_multi_tenant_with_swift_config(self):
    """
        Test that Glance does not start when a config file is set on
        multi-tenant mode
        """
    schemes = ['swift', 'swift+config']
    for s in schemes:
        self.config(group='glance_store', default_backend='swift1')
        self.config(group='swift1', swift_store_config_file='not/none', swift_store_multi_tenant=True)
        self.assertRaises(exceptions.BadStoreConfiguration, Store, self.conf, backend='swift1')