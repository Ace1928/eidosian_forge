import collections
from osc_lib.cli import format_columns
from osc_lib.tests import utils
def test_list_dict_column(self):
    data = [{'key1': 'value1'}, {'key2': 'value2'}]
    col = format_columns.ListDictColumn(data)
    self.assertEqual(data, col.machine_readable())
    self.assertEqual("key1='value1'\nkey2='value2'", col.human_readable())