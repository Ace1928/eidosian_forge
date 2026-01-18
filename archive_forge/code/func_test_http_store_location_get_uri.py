from unittest import mock
import requests
import glance_store
from glance_store._drivers import http
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
def test_http_store_location_get_uri(self):
    """Test for HTTP URI with and without query"""
    uris = ['http://netloc/path/to/file.tar.gzhttp://netloc/path/to/file.tar.gz?query=text']
    for uri in uris:
        loc = location.get_location_from_uri(uri, conf=self.conf)
        self.assertEqual(uri, loc.store_location.get_uri())