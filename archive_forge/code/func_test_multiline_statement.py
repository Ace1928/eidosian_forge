import pytest
from IPython.utils.tokenutil import token_at_cursor, line_at_cursor
@pytest.mark.parametrize('c, token', zip(list(range(16, 22)) + list(range(22, 28)), ['int'] * (22 - 16) + ['map'] * (28 - 22)))
def test_multiline_statement(c, token):
    cell = 'a = (1,\n    3)\n\nint()\nmap()\n'
    expect_token(token, cell, c)