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
def test_download_forbidden(self):
    """Make sure the serializer can return 403 forbidden error instead of
        500 internal server error.
        """

    def get_data(*args, **kwargs):
        raise exception.Forbidden()
    self.mock_object(glance.domain.proxy.Image, 'get_data', get_data)
    request = wsgi.Request.blank('/')
    request.environ = {}
    response = webob.Response()
    response.request = request
    image = FakeImage(size=3, data=iter('ZZZ'))
    image.get_data = get_data
    self.assertRaises(webob.exc.HTTPForbidden, self.serializer.download, response, image)