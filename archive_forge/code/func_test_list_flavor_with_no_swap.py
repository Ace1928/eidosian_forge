from novaclient.tests.functional import base
def test_list_flavor_with_no_swap(self):
    self._create_flavor()
    out = self.nova('flavor-list')
    self.assertEqual(self.SWAP_DEFAULT, self._get_column_value_from_single_row_table(out, 'Swap'))