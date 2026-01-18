import glance.api.v2.schemas
import glance.db.sqlalchemy.api as db_api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_images(self):
    req = unit_test_utils.get_fake_request()
    output = self.controller.images(req)
    self.assertEqual('images', output['name'])
    expected = set(['images', 'schema', 'first', 'next'])
    self.assertEqual(expected, set(output['properties'].keys()))
    expected = set(['{schema}', '{first}', '{next}'])
    actual = set([link['href'] for link in output['links']])
    self.assertEqual(expected, actual)