import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_create_forbidden_by_add_policy(self):
    rules = {'add_member': False}
    self.policy.set_rules(rules)
    request = unit_test_utils.get_fake_request()
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.create, request, image_id=UUID2, member_id=TENANT3)