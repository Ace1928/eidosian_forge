from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
def test_filter_strips_extra_properties(self):
    obj = {'ham': 'virginia', 'eggs': 'scrambled', 'bacon': 'crispy'}
    filtered = self.schema.filter(obj)
    expected = {'ham': 'virginia', 'eggs': 'scrambled'}
    self.assertEqual(expected, filtered)