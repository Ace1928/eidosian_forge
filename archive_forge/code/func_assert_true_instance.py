import functools
import re
import tokenize
from hacking import core
@skip_ignored_lines
@core.flake8ext
def assert_true_instance(logical_line, filename):
    """Check for assertTrue(isinstance(a, b)) sentences

    N320
    """
    if re_assert_true_instance.match(logical_line):
        yield (0, 'N320 assertTrue(isinstance(a, b)) sentences not allowed, you should use assertIsInstance(a, b) instead.')