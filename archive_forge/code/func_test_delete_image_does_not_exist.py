import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_delete_image_does_not_exist(self):
    request = unit_test_utils.get_fake_request()
    member_id = TENANT2
    image_id = 'fake-image-id'
    self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete, request, image_id, member_id)