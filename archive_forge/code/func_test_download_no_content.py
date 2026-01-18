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
def test_download_no_content(self):
    """Test image download returns HTTPNoContent

        Make sure that serializer returns 204 no content error in case of
        image data is not available at specified location.
        """
    with mock.patch.object(glance.domain.proxy.Image, 'get_data') as mock_get_data:
        mock_get_data.side_effect = glance_store.NotFound(image='image')
        request = wsgi.Request.blank('/')
        response = webob.Response()
        response.request = request
        image = FakeImage(size=3, data=iter('ZZZ'))
        image.get_data = mock_get_data
        self.assertRaises(webob.exc.HTTPNoContent, self.serializer.download, response, image)