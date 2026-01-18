import http.client as http
import io
from unittest import mock
import uuid
from cursive import exception as cursive_exception
import glance_store
from glance_store._drivers import filesystem
from oslo_config import cfg
import webob
import glance.api.policy
import glance.api.v2.image_data
from glance.common import exception
from glance.common import wsgi
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_download_failure_with_valid_range(self):
    with mock.patch.object(glance.domain.proxy.Image, 'get_data') as mock_get_data:
        mock_get_data.side_effect = glance_store.NotFound(image='image')
    request = wsgi.Request.blank('/')
    request.environ = {}
    request.headers['Range'] = 'bytes=1-2'
    response = webob.Response()
    response.request = request
    image = FakeImage(size=3, data=[b'Z', b'Z', b'Z'])
    image.get_data = mock_get_data
    self.assertRaises(webob.exc.HTTPNoContent, self.serializer.download, response, image)