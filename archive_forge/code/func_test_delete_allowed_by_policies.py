import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_delete_allowed_by_policies(self):
    rules = {'get_member': True, 'delete_member': True}
    self.policy.set_rules(rules)
    request = unit_test_utils.get_fake_request(tenant=TENANT1)
    output = self.controller.delete(request, image_id=UUID2, member_id=TENANT4)
    request = unit_test_utils.get_fake_request()
    output = self.controller.index(request, UUID2)
    self.assertEqual(0, len(output['members']))