import pytest
from wasabi.util import color, diff_strings, format_repr, locale_escape, wrap
def test_diff_strings_with_symbols():
    a = 'hello\nworld\nwide\nweb'
    b = 'yo\nwide\nworld\nweb'
    expected = '\x1b[38;5;16;48;5;2m+ yo\x1b[0m\n\x1b[38;5;16;48;5;2m+ wide\x1b[0m\n\x1b[38;5;16;48;5;1m- hello\x1b[0m\nworld\n\x1b[38;5;16;48;5;1m- wide\x1b[0m\nweb'
    assert diff_strings(a, b, add_symbols=True) == expected