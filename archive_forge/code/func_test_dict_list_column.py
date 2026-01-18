import collections
from osc_lib.cli import format_columns
from osc_lib.tests import utils
def test_dict_list_column(self):
    data = {'public': ['2001:db8::8', '172.24.4.6'], 'private': ['2000:db7::7', '192.24.4.6']}
    col = format_columns.DictListColumn(data)
    self.assertEqual(data, col.machine_readable())
    self.assertEqual('private=192.24.4.6, 2000:db7::7; public=172.24.4.6, 2001:db8::8', col.human_readable())