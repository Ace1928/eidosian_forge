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
def test_init_client_multi_tenant_insecure(self):
    """
        Test that keystone client was initialized correctly with no
        certificate verification.
        """
    with mock.patch.object(swift.MultiTenantStore, '_set_url_prefix'):
        self._init_client(verify=False, swift_store_multi_tenant=True, swift_store_auth_insecure=True, swift_store_config_file=None)