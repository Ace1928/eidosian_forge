from troveclient.osc.v1 import database_flavors
from troveclient.tests.osc.v1 import fakes
def test_flavor_list_with_optional_args(self):
    args = ['--datastore-type', 'mysql', '--datastore-version-id', '5.6']
    parsed_args = self.check_parser(self.cmd, args, [])
    list_flavor_dict = {'datastore': 'mysql', 'version_id': '5.6'}
    columns, values = self.cmd.take_action(parsed_args)
    self.flavor_client.list_datastore_version_associated_flavors.assert_called_once_with(**list_flavor_dict)
    self.assertEqual(self.columns, columns)
    self.assertEqual([self.values], values)