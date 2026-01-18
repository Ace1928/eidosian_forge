from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_delete_wb_from_another_tenant(self):
    wb = self.workbook_create(self.wb_def)
    name = self.get_field_value(wb, 'Name')
    self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'workbook-delete', params=name)