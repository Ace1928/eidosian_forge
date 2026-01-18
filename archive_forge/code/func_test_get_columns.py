from osc_lib.tests import utils as test_utils
from osc_lib.utils import columns as column_utils
def test_get_columns(self):
    item = {'id': 'test-id', 'tenant_id': 'test-tenant_id', 'foo': 'bar'}
    attr_map = (('id', 'ID', column_utils.LIST_BOTH), ('tenant_id', 'Project', column_utils.LIST_LONG_ONLY), ('name', 'Name', column_utils.LIST_BOTH))
    columns, display_names = column_utils.get_columns(item, attr_map)
    self.assertEqual(tuple(['id', 'tenant_id', 'foo']), columns)
    self.assertEqual(tuple(['ID', 'Project', 'foo']), display_names)