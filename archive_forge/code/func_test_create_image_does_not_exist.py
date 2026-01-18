import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_create_image_does_not_exist(self):
    request = unit_test_utils.get_fake_request()
    image_id = 'fake-image-id'
    member_id = TENANT3
    self.assertRaises(webob.exc.HTTPNotFound, self.controller.create, request, image_id=image_id, member_id=member_id)