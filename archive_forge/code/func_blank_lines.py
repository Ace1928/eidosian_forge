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
def blank_lines(logical_line, blank_lines, indent_level, line_number, blank_before, previous_logical, previous_unindented_logical_line, previous_indent_level, lines):
    """Separate top-level function and class definitions with two blank lines.

    Method definitions inside a class are separated by a single blank line.

    Extra blank lines may be used (sparingly) to separate groups of related
    functions.  Blank lines may be omitted between a bunch of related
    one-liners (e.g. a set of dummy implementations).

    Use blank lines in functions, sparingly, to indicate logical sections.

    Okay: def a():\\n    pass\\n\\n\\ndef b():\\n    pass
    Okay: def a():\\n    pass\\n\\n\\nasync def b():\\n    pass
    Okay: def a():\\n    pass\\n\\n\\n# Foo\\n# Bar\\n\\ndef b():\\n    pass
    Okay: default = 1\\nfoo = 1
    Okay: classify = 1\\nfoo = 1

    E301: class Foo:\\n    b = 0\\n    def bar():\\n        pass
    E302: def a():\\n    pass\\n\\ndef b(n):\\n    pass
    E302: def a():\\n    pass\\n\\nasync def b(n):\\n    pass
    E303: def a():\\n    pass\\n\\n\\n\\ndef b(n):\\n    pass
    E303: def a():\\n\\n\\n\\n    pass
    E304: @decorator\\n\\ndef a():\\n    pass
    E305: def a():\\n    pass\\na()
    """
    if line_number < 3 and (not previous_logical):
        return
    if previous_logical.startswith('@'):
        if blank_lines:
            yield (0, 'E304 blank lines found after function decorator')
    elif blank_lines > 2 or (indent_level and blank_lines == 2):
        yield (0, 'E303 too many blank lines (%d)' % blank_lines)
    elif logical_line.startswith(('def ', 'async def', 'class ', '@')):
        if indent_level:
            if not (blank_before or previous_indent_level < indent_level or DOCSTRING_REGEX.match(previous_logical)):
                ancestor_level = indent_level
                nested = False
                for line in lines[line_number - 2::-1]:
                    if line.strip() and expand_indent(line) < ancestor_level:
                        ancestor_level = expand_indent(line)
                        nested = line.lstrip().startswith('def ')
                        if nested or ancestor_level == 0:
                            break
                if nested:
                    yield (0, 'E306 expected 1 blank line before a nested definition, found 0')
                else:
                    yield (0, 'E301 expected 1 blank line, found 0')
        elif blank_before != 2:
            yield (0, 'E302 expected 2 blank lines, found %d' % blank_before)
    elif logical_line and (not indent_level) and (blank_before != 2) and previous_unindented_logical_line.startswith(('def ', 'class ')):
        yield (0, 'E305 expected 2 blank lines after class or function definition, found %d' % blank_before)