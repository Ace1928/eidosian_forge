from keystone.common import driver_hints
from keystone.tests.unit import core as test
def test_multiple_creates(self):
    hints = driver_hints.Hints()
    hints.add_filter('t1', 'data1')
    hints.add_filter('t2', 'data2')
    self.assertEqual(2, len(hints.filters))
    hints2 = driver_hints.Hints()
    hints2.add_filter('t4', 'data1')
    hints2.add_filter('t5', 'data2')
    self.assertEqual(2, len(hints.filters))