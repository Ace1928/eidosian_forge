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
def test_download_store_random_get_not_support(self):
    """Test image download returns HTTPBadRequest.

        Make sure that serializer returns 400 bad request error in case of
        getting randomly images from this store is not supported at
        specified location.
        """
    with mock.patch.object(glance.domain.proxy.Image, 'get_data') as m_get_data:
        err = glance_store.StoreRandomGetNotSupported(offset=0, chunk_size=0)
        m_get_data.side_effect = err
        request = wsgi.Request.blank('/')
        response = webob.Response()
        response.request = request
        image = FakeImage(size=3, data=iter('ZZZ'))
        image.get_data = m_get_data
        self.assertRaises(webob.exc.HTTPBadRequest, self.serializer.download, response, image)