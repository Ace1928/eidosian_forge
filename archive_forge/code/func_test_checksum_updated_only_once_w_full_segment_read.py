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
def test_checksum_updated_only_once_w_full_segment_read(self):
    expected_csum = md5(usedforsecurity=False)
    expected_csum.update(b'1234567')
    expected_multihash = hashlib.sha256()
    expected_multihash.update(b'1234567')
    self.reader.read(7)
    self.reader.seek(4)
    self.reader.read(1)
    self.assertEqual(expected_csum.hexdigest(), self.checksum.hexdigest())
    self.assertEqual(expected_multihash.hexdigest(), self.os_hash_value.hexdigest())