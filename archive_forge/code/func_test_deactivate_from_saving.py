import glance_store as store
import webob
import glance.api.v2.image_actions as image_actions
import glance.context
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_deactivate_from_saving(self):
    self._test_deactivate_from_wrong_status('saving')