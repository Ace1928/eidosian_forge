import glance_store as store
import webob
import glance.api.v2.image_actions as image_actions
import glance.context
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_deactivate_from_active(self):
    self._create_image('active')
    request = unit_test_utils.get_fake_request()
    self.controller.deactivate(request, UUID1)
    image = self.db.image_get(request.context, UUID1)
    self.assertEqual('deactivated', image['status'])