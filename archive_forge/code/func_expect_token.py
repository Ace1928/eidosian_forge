import pytest
from IPython.utils.tokenutil import token_at_cursor, line_at_cursor
def expect_token(expected, cell, cursor_pos):
    token = token_at_cursor(cell, cursor_pos)
    offset = 0
    for line in cell.splitlines():
        if offset + len(line) >= cursor_pos:
            break
        else:
            offset += len(line) + 1
    column = cursor_pos - offset
    line_with_cursor = '%s|%s' % (line[:column], line[column:])
    assert token == expected, 'Expected %r, got %r in: %r (pos %i)' % (expected, token, line_with_cursor, cursor_pos)