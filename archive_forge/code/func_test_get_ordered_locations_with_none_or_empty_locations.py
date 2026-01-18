import copy
import stevedore
from glance.common import location_strategy
from glance.common.location_strategy import location_order
from glance.common.location_strategy import store_type
from glance.tests.unit import base
def test_get_ordered_locations_with_none_or_empty_locations(self):
    self.assertEqual([], location_strategy.get_ordered_locations(None))
    self.assertEqual([], location_strategy.get_ordered_locations([]))