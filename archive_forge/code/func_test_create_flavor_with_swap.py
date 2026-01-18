from novaclient.tests.functional import base
def test_create_flavor_with_swap(self):
    out, _ = self._create_flavor(swap=10)
    self.assertEqual('10', self._get_column_value_from_single_row_table(out, 'Swap'))