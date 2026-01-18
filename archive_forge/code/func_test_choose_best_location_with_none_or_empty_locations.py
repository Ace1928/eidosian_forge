import copy
import stevedore
from glance.common import location_strategy
from glance.common.location_strategy import location_order
from glance.common.location_strategy import store_type
from glance.tests.unit import base
def test_choose_best_location_with_none_or_empty_locations(self):
    self.assertIsNone(location_strategy.choose_best_location(None))
    self.assertIsNone(location_strategy.choose_best_location([]))