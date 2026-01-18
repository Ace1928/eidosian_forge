from heat.scaling import scalingutil as sc_util
from heat.tests import common
def test_calc(self):
    self.assertEqual(self.expected, sc_util.calculate_new_capacity(self.current, self.adjustment, self.adjustment_type, self.min_adjustment_step, self.minimum, self.maximum))