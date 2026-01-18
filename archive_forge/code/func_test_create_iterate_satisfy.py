from keystone.common import driver_hints
from keystone.tests.unit import core as test
def test_create_iterate_satisfy(self):
    hints = driver_hints.Hints()
    hints.add_filter('t1', 'data1')
    hints.add_filter('t2', 'data2')
    self.assertEqual(2, len(hints.filters))
    filter = hints.get_exact_filter_by_name('t1')
    self.assertEqual('t1', filter['name'])
    self.assertEqual('data1', filter['value'])
    self.assertEqual('equals', filter['comparator'])
    self.assertFalse(filter['case_sensitive'])
    hints.filters.remove(filter)
    filter_count = 0
    for filter in hints.filters:
        filter_count += 1
        self.assertEqual('t2', filter['name'])
    self.assertEqual(1, filter_count)