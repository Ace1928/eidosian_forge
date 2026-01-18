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
from glance_store._drivers.swift import buffered
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import backend
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_get_using_slice(self):
    """Test a "normal" retrieval of an image in chunks."""
    uri = 'swift://%s:key@auth_address/glance/%s' % (self.swift_store_user, FAKE_UUID)
    loc = location.get_location_from_uri(uri, conf=self.conf)
    image_swift, image_size = self.store.get(loc)
    self.assertEqual(5120, image_size)
    expected_data = b'*' * FIVE_KB
    self.assertEqual(expected_data, image_swift[:])
    expected_data = b'*' * (FIVE_KB - 100)
    self.assertEqual(expected_data, image_swift[100:])