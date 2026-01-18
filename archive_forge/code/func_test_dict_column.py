import collections
from osc_lib.cli import format_columns
from osc_lib.tests import utils
def test_dict_column(self):
    data = {'key1': 'value1', 'key2': 'value2'}
    col = format_columns.DictColumn(data)
    self.assertEqual(data, col.machine_readable())
    self.assertEqual("key1='value1', key2='value2'", col.human_readable())