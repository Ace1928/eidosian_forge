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
def test_get_container_name_with_dash(self):
    self.config(group='swift1', swift_store_multiple_containers_seed=10)
    self.store = swift.SingleTenantStore(self.conf, backend='swift1')
    test_image_id = 'fdae39a1-bac5-4238-aba4-69bcc726e848'
    actual = self.store.get_container_name(test_image_id, 'default_container')
    expected = 'default_container_' + 'fdae39a1-ba'
    self.assertEqual(expected, actual)