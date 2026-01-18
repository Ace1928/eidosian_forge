import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
def join_continuation(lines):
    """
    Join lines continued by a trailing backslash.

    >>> list(join_continuation(['foo \\\\', 'bar', 'baz']))
    ['foobar', 'baz']
    >>> list(join_continuation(['foo \\\\', 'bar', 'baz']))
    ['foobar', 'baz']
    >>> list(join_continuation(['foo \\\\', 'bar \\\\', 'baz']))
    ['foobarbaz']

    Not sure why, but...
    The character preceding the backslash is also elided.

    >>> list(join_continuation(['goo\\\\', 'dly']))
    ['godly']

    A terrible idea, but...
    If no line is available to continue, suppress the lines.

    >>> list(join_continuation(['foo', 'bar\\\\', 'baz\\\\']))
    ['foo']
    """
    lines = iter(lines)
    for item in lines:
        while item.endswith('\\'):
            try:
                item = item[:-2].strip() + next(lines)
            except StopIteration:
                return
        yield item