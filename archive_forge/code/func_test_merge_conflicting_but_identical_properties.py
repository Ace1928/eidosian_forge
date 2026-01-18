from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
def test_merge_conflicting_but_identical_properties(self):
    conflicts = {'ham': {'type': 'string'}}
    self.schema.merge_properties(conflicts)
    expected = set(['ham', 'eggs'])
    actual = set(self.schema.raw()['properties'].keys())
    self.assertEqual(expected, actual)