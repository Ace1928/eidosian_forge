import http.client as http
import webob
import glance.api.v2.image_tags
from glance.common import exception
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.unit.v2.test_image_data_resource as image_data_tests
import glance.tests.utils as test_utils
def test_delete_tag_forbidden(self):

    def fake_get(self):
        raise exception.Forbidden()
    image_repo = image_data_tests.FakeImageRepo()
    image_repo.get = fake_get

    def get_fake_repo(self):
        return image_repo
    self.controller.gateway.get_repo = get_fake_repo
    request = unit_test_utils.get_fake_request()
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, unit_test_utils.UUID1, 'ping')