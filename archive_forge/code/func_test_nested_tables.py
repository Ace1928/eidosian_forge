import ddt
from manilaclient.tests.functional import utils as func_utils
from manilaclient.tests.unit import utils
@ddt.data({'input': OUTPUT_LINES_NESTED_TABLE, 'valid_nested': {'headers': ['aa', 'bb'], 'values': []}}, {'input': OUTPUT_LINES_NESTED_TABLE_MULTI_LINE, 'valid_nested': {'headers': ['id', 'bb'], 'values': [['01', ['a1', 'a2']]]}})
@ddt.unpack
def test_nested_tables(self, input, valid_nested):
    actual_result = func_utils.multi_line_row_table(input, group_by_column_index=1)
    self.assertEqual(['**', 'ID', 'Name', 'Status'], actual_result['headers'])
    self.assertEqual(2, len(actual_result['values']))
    self.assertEqual(valid_nested, actual_result['values'][0][3])