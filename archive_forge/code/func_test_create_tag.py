import http.client as http
import webob
import glance.api.v2.image_tags
from glance.common import exception
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.unit.v2.test_image_data_resource as image_data_tests
import glance.tests.utils as test_utils
def test_create_tag(self):
    response = webob.Response()
    self.serializer.update(response, None)
    self.assertEqual(http.NO_CONTENT, response.status_int)