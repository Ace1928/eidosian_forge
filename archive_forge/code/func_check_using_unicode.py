import functools
import re
import tokenize
from hacking import core
@skip_ignored_lines
@core.flake8ext
def check_using_unicode(logical_line, filename):
    """Check crosspython unicode usage

    N353
    """
    if re.search('\\bunicode\\(', logical_line):
        yield (0, "N353 'unicode' function is absent in python3. Please use 'str' instead.")