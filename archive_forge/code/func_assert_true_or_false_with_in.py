import functools
import re
import tokenize
from hacking import core
@skip_ignored_lines
@core.flake8ext
def assert_true_or_false_with_in(logical_line, filename):
    """Check assertTrue/False(A in/not in B) with collection contents

    Check for assertTrue/False(A in B), assertTrue/False(A not in B),
    assertTrue/False(A in B, message) or assertTrue/False(A not in B, message)
    sentences.

    N323
    """
    res = re_assert_true_false_with_in_or_not_in.search(logical_line) or re_assert_true_false_with_in_or_not_in_spaces.search(logical_line)
    if res:
        yield (0, 'N323 assertTrue/assertFalse(A in/not in B)sentences not allowed, you should use assertIn(A, B) or assertNotIn(A, B) instead.')