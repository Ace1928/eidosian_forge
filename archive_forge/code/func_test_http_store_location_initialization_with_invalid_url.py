from unittest import mock
import requests
import glance_store
from glance_store._drivers import http
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
def test_http_store_location_initialization_with_invalid_url(self):
    """Test store location initialization from incorrect uris."""
    incorrect_uris = ['http://127.0.0.1:~/ubuntu.iso', 'http://openstack.com:some_text/ubuntu.iso', 'http://[1080::8:800:200C:417A]:some_text/ubuntu.iso']
    for uri in incorrect_uris:
        self.assertRaises(exceptions.BadStoreUri, location.get_location_from_uri, uri)