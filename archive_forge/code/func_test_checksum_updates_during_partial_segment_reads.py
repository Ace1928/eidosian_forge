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
def test_checksum_updates_during_partial_segment_reads(self):
    expected_csum = md5(usedforsecurity=False)
    expected_multihash = hashlib.sha256()
    self.reader.read(4)
    expected_csum.update(b'1234')
    expected_multihash.update(b'1234')
    self.assertEqual(expected_csum.hexdigest(), self.checksum.hexdigest())
    self.assertEqual(expected_multihash.hexdigest(), self.os_hash_value.hexdigest())
    self.reader.seek(0)
    self.reader.read(2)
    self.assertEqual(expected_csum.hexdigest(), self.checksum.hexdigest())
    self.assertEqual(expected_multihash.hexdigest(), self.os_hash_value.hexdigest())
    self.reader.read(4)
    expected_csum.update(b'56')
    expected_multihash.update(b'56')
    self.assertEqual(expected_csum.hexdigest(), self.checksum.hexdigest())
    self.assertEqual(expected_multihash.hexdigest(), self.os_hash_value.hexdigest())