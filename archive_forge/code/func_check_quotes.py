import functools
import re
import tokenize
from hacking import core
@skip_ignored_lines
@core.flake8ext
def check_quotes(logical_line, filename):
    """Check that single quotation marks are not used

    N350
    """
    in_string = False
    in_multiline_string = False
    single_quotas_are_used = False
    check_tripple = lambda line, i, char: i + 2 < len(line) and char == line[i] == line[i + 1] == line[i + 2]
    i = 0
    while i < len(logical_line):
        char = logical_line[i]
        if in_string:
            if char == '"':
                in_string = False
            if char == '\\':
                i += 1
        elif in_multiline_string:
            if check_tripple(logical_line, i, '"'):
                i += 2
                in_multiline_string = False
        elif char == '#':
            break
        elif char == "'":
            single_quotas_are_used = True
            break
        elif char == '"':
            if check_tripple(logical_line, i, '"'):
                in_multiline_string = True
                i += 3
                continue
            in_string = True
        i += 1
    if single_quotas_are_used:
        yield (i, 'N350 Remove Single quotes')