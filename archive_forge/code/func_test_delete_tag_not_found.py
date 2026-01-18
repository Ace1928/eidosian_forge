import http.client as http
import webob
import glance.api.v2.image_tags
from glance.common import exception
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.unit.v2.test_image_data_resource as image_data_tests
import glance.tests.utils as test_utils
def test_delete_tag_not_found(self):
    request = unit_test_utils.get_fake_request()
    self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete, request, unit_test_utils.UUID1, 'what')