import re
from hacking import core
@core.flake8ext
def no_xrange(logical_line):
    """Disallow 'xrange()'

    O340
    """
    if assert_no_xrange_re.match(logical_line):
        yield (0, 'O340: Do not use xrange().')