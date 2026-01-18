from unittest import mock
import requests
import glance_store
from glance_store._drivers import http
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
def test_http_add_raise_error(self):
    self.assertRaises(exceptions.StoreAddDisabled, self.store.add, None, None, None, None)
    self.assertRaises(exceptions.StoreAddDisabled, glance_store.add_to_backend, None, None, None, None, 'http')