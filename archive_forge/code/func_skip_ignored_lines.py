import functools
import re
import tokenize
from hacking import core
@core.flake8ext
def skip_ignored_lines(func):

    @functools.wraps(func)
    def wrapper(logical_line, filename):
        line = logical_line.strip()
        if not line or line.startswith('#') or line.endswith('# noqa'):
            return
        try:
            yield next(func(logical_line, filename))
        except StopIteration:
            return
    return wrapper