from unittest import mock
import requests
import glance_store
from glance_store._drivers import http
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
def test_http_store_location_initialization(self):
    """Test store location initialization from valid uris"""
    uris = ['http://127.0.0.1:8000/ubuntu.iso', 'http://openstack.com:80/ubuntu.iso', 'http://[1080::8:800:200C:417A]:80/ubuntu.iso']
    for uri in uris:
        location.get_location_from_uri(uri)