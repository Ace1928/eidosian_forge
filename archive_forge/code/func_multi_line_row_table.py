import ast
from tempest.lib.cli import output_parser
import testtools
from manilaclient import api_versions
from manilaclient import config
def multi_line_row_table(output_lines, group_by_column_index=0):
    parsed_table = output_parser.table(output_lines)
    rows = parsed_table['values']
    row_index = 0

    def get_column_index(column_name, headers, default):
        return next((i for i, h in enumerate(headers) if h.lower() == column_name), default)
    if group_by_column_index is None:
        group_by_column_index = get_column_index('id', parsed_table['headers'], 0)

    def is_embedded_table(parsed_rows):

        def is_table_border(t):
            return str(t).startswith('+')
        return isinstance(parsed_rows, list) and len(parsed_rows) > 3 and is_table_border(parsed_rows[0]) and is_table_border(parsed_rows[-1])

    def merge_cells(master_cell, value_cell):
        if value_cell:
            if not isinstance(master_cell, list):
                master_cell = [master_cell]
            master_cell.append(value_cell)
        if is_embedded_table(master_cell):
            return multi_line_row_table('\n'.join(master_cell), None)
        return master_cell

    def is_empty_row(row):
        empty_cells = 0
        for cell in row:
            if cell == '':
                empty_cells += 1
        return len(row) == empty_cells
    while row_index < len(rows):
        row = rows[row_index]
        line_with_value = row_index > 0 and row[group_by_column_index] == ''
        if line_with_value and (not is_empty_row(row)):
            rows[row_index - 1] = list(map(merge_cells, rows[row_index - 1], rows.pop(row_index)))
        else:
            row_index += 1
    return parsed_table