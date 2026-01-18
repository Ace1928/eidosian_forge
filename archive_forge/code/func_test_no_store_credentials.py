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
def test_no_store_credentials(self):
    """
        Tests that options without a valid credentials disables the add method
        """
    self.store = Store(self.conf, backend='swift1')
    self.store.ref_params = {'ref1': {'auth_address': 'authurl.com', 'user': '', 'key': ''}}
    self.store.configure()
    self.assertFalse(self.store.is_capable(capabilities.BitMasks.WRITE_ACCESS))