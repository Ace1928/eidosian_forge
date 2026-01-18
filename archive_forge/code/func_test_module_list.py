from openstackclient.tests.functional import base
def test_module_list(self):
    cmd_output = self.openstack('module list', parse_output=True)
    for one_module in self.CLIENTS:
        self.assertIn(one_module, cmd_output.keys())
    for one_module in self.LIBS:
        self.assertNotIn(one_module, cmd_output.keys())
    cmd_output = self.openstack('module list --all', parse_output=True)
    for one_module in self.CLIENTS + self.LIBS:
        self.assertIn(one_module, cmd_output.keys())