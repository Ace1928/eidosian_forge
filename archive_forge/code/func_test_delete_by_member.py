import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_delete_by_member(self):
    enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'delete_member': "'{0}':%(owner)s".format(TENANT4), 'get_members': '', 'get_member': ''})
    request = unit_test_utils.get_fake_request(tenant=TENANT4)
    self.controller.policy = enforcer
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete, request, UUID2, TENANT4)
    request = unit_test_utils.get_fake_request()
    output = self.controller.index(request, UUID2)
    self.assertEqual(1, len(output['members']))
    actual = set([image_member.member_id for image_member in output['members']])
    expected = set([TENANT4])
    self.assertEqual(expected, actual)