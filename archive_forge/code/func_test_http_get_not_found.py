from unittest import mock
import requests
import glance_store
from glance_store._drivers import http
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
def test_http_get_not_found(self):
    self._mock_requests()
    fake = utils.fake_response(status_code=404, content='404 Not Found')
    self.request.return_value = fake
    uri = 'http://netloc/path/to/file.tar.gz'
    loc = location.get_location_from_uri(uri, conf=self.conf)
    self.assertRaises(exceptions.NotFound, self.store.get, loc)