from unittest import mock
import requests
import glance_store
from glance_store._drivers import http
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
def test_http_get_size_with_non_existent_image_raises_Not_Found(self):
    self._mock_requests()
    self.request.return_value = utils.fake_response(status_code=404, content='404 Not Found')
    uri = 'http://netloc/path/to/file.tar.gz'
    loc = location.get_location_from_uri(uri, conf=self.conf)
    self.assertRaises(exceptions.NotFound, self.store.get_size, loc)
    self.request.assert_called_once_with('HEAD', uri, stream=True, allow_redirects=False)