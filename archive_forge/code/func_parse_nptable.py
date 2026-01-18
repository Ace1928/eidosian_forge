import re
from ..helpers import PREVENT_BACKSLASH
def parse_nptable(block, m, state):
    header = m.group('nptable_head')
    align = m.group('nptable_align')
    thead, aligns = _process_thead(header, align)
    if not thead:
        return
    rows = []
    body = m.group('nptable_body')
    for text in body.splitlines():
        row = _process_row(text, aligns)
        if not row:
            return
        rows.append(row)
    children = [thead, {'type': 'table_body', 'children': rows}]
    state.append_token({'type': 'table', 'children': children})
    return m.end()