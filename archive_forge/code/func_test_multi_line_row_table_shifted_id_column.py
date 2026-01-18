import ddt
from manilaclient.tests.functional import utils as func_utils
from manilaclient.tests.unit import utils
def test_multi_line_row_table_shifted_id_column(self):
    input = self.OUTPUT_LINES_COMPLICATED_MULTI_ROW_WITH_SHIFTED_ID
    valid_values = [['**', '11', 'foo', 'BUILD'], ['', '21', 'bar', ['ERROR', 'ERROR2', 'ERROR3']], ['', '', '', ''], ['**', '31', ['bee', 'bee2'], 'None'], ['', '', '', '']]
    actual_result = func_utils.multi_line_row_table(input, group_by_column_index=1)
    self.assertEqual(['**', 'ID', 'Name', 'Status'], actual_result['headers'])
    self.assertEqual(valid_values, actual_result['values'])