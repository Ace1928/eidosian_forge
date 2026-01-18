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
def test_connection_with_conf_endpoint(self):
    ctx = mock.MagicMock(user='tenant:user1', tenant='tenant')
    self.config(group='swift1', swift_store_endpoint='https://internal.com')
    self.store.configure()
    connection = self.store.get_connection(self.location, context=ctx)
    self.assertEqual('https://example.com/v2/', connection.authurl)
    self.assertEqual('2', connection.auth_version)
    self.assertEqual('user1', connection.user)
    self.assertEqual('tenant', connection.tenant_name)
    self.assertEqual('key1', connection.key)
    self.assertEqual('https://internal.com', connection.preauthurl)
    self.assertFalse(connection.insecure)
    self.assertEqual({'service_type': 'object-store', 'endpoint_type': 'publicURL'}, connection.os_options)