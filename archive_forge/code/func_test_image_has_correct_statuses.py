import glance.api.v2.schemas
import glance.db.sqlalchemy.api as db_api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_image_has_correct_statuses(self):
    req = unit_test_utils.get_fake_request()
    output = self.controller.image(req)
    self.assertEqual('image', output['name'])
    expected_statuses = set(db_api.STATUSES)
    actual_statuses = set(output['properties']['status']['enum'])
    self.assertEqual(expected_statuses, actual_statuses)