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
def test_light_buffer(self):
    s = b'12'
    infile = io.BytesIO(s)
    infile.seek(0)
    total = 7
    checksum = md5(usedforsecurity=False)
    os_hash_value = hashlib.sha256()
    self.reader = buffered.BufferedReader(infile, checksum, os_hash_value, total)
    self.reader.read(0)
    self.assertEqual(b'12', self.reader.read(7))
    self.assertEqual(2, self.reader.tell())