import functools
import re
import tokenize
from hacking import core
@skip_ignored_lines
@core.flake8ext
def assert_equal_type(logical_line, filename):
    """Check for assertEqual(type(A), B) sentences

    N321
    """
    if re_assert_equal_type.match(logical_line):
        yield (0, 'N321 assertEqual(type(A), B) sentences not allowed, you should use assertIsInstance(a, b) instead.')