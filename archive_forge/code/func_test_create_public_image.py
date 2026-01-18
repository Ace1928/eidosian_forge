import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_create_public_image(self):
    request = unit_test_utils.get_fake_request()
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.create, request, UUID1, TENANT2)