from novaclient.tests.functional import base
from novaclient.tests.functional.v2 import fake_crypto
from novaclient.tests.functional.v2.legacy import test_keypairs
def test_create_and_list_keypair_with_marker_and_limit(self):
    names = []
    for i in range(3):
        names.append(self.name_generate())
        self.nova('keypair-add %s --user %s' % (names[i], self.user_id))
        self.addCleanup(self.another_nova, 'keypair-delete %s' % names[i])
    names = sorted(names)
    output_1 = self.another_nova('keypair-list --limit 1 --marker %s' % names[0])
    output_2 = self.nova('keypair-list --limit 1 --marker %s --user %s' % (names[0], self.user_id))
    self.assertEqual(output_1, output_2)
    self.assertEqual(names[1], self._get_column_value_from_single_row_table(output_1, 'Name'))