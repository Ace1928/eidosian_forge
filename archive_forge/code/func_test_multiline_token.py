import pytest
from IPython.utils.tokenutil import token_at_cursor, line_at_cursor
def test_multiline_token():
    cell = '\n'.join(['"""\n\nxxxxxxxxxx\n\n"""', '5, """', 'docstring', 'multiline token', '""", [', '2, 3, "complicated"]', 'b = hello("string", there)'])
    expected = 'hello'
    start = cell.index(expected) + 1
    for i in range(start, start + len(expected)):
        expect_token(expected, cell, i)
    expected = 'hello'
    start = cell.index(expected) + 1
    for i in range(start, start + len(expected)):
        expect_token(expected, cell, i)