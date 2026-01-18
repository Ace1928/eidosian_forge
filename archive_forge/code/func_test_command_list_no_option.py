from openstackclient.tests.functional import base
def test_command_list_no_option(self):
    cmd_output = self.openstack('command list', parse_output=True)
    group_names = [each.get('Command Group') for each in cmd_output]
    for one_group in self.GROUPS:
        self.assertIn(one_group, group_names)