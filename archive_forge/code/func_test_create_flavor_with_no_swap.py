from novaclient.tests.functional import base
def test_create_flavor_with_no_swap(self):
    out, _ = self._create_flavor()
    self.assertEqual(self.SWAP_DEFAULT, self._get_column_value_from_single_row_table(out, 'Swap'))