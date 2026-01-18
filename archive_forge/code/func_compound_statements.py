from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def compound_statements(logical_line):
    """Compound statements (on the same line) are generally discouraged.

    While sometimes it's okay to put an if/for/while with a small body
    on the same line, never do this for multi-clause statements.
    Also avoid folding such long lines!

    Always use a def statement instead of an assignment statement that
    binds a lambda expression directly to a name.

    Okay: if foo == 'blah':\\n    do_blah_thing()
    Okay: do_one()
    Okay: do_two()
    Okay: do_three()

    E701: if foo == 'blah': do_blah_thing()
    E701: for x in lst: total += x
    E701: while t < 10: t = delay()
    E701: if foo == 'blah': do_blah_thing()
    E701: else: do_non_blah_thing()
    E701: try: something()
    E701: finally: cleanup()
    E701: if foo == 'blah': one(); two(); three()
    E702: do_one(); do_two(); do_three()
    E703: do_four();  # useless semicolon
    E704: def f(x): return 2*x
    E731: f = lambda x: 2*x
    """
    line = logical_line
    last_char = len(line) - 1
    found = line.find(':')
    prev_found = 0
    counts = dict(((char, 0) for char in '{}[]()'))
    while -1 < found < last_char:
        update_counts(line[prev_found:found], counts)
        if counts['{'] <= counts['}'] and counts['['] <= counts[']'] and (counts['('] <= counts[')']):
            lambda_kw = LAMBDA_REGEX.search(line, 0, found)
            if lambda_kw:
                before = line[:lambda_kw.start()].rstrip()
                if before[-1:] == '=' and isidentifier(before[:-1].strip()):
                    yield (0, 'E731 do not assign a lambda expression, use a def')
                break
            if line.startswith('def '):
                yield (0, 'E704 multiple statements on one line (def)')
            else:
                yield (found, 'E701 multiple statements on one line (colon)')
        prev_found = found
        found = line.find(':', found + 1)
    found = line.find(';')
    while -1 < found:
        if found < last_char:
            yield (found, 'E702 multiple statements on one line (semicolon)')
        else:
            yield (found, 'E703 statement ends with a semicolon')
        found = line.find(';', found + 1)