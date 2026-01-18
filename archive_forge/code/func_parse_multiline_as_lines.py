import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
def parse_multiline_as_lines(s):
    """Same as parse_multiline, but returns a list of lines.

    (This is the inverse of format_multiline_lines.)
    """
    lines = s.splitlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if line.startswith(' '):
            line = line[1:]
        else:
            raise MachineReadableFormatError('continued line must begin with " "')
        if line == '.':
            line = ''
        lines[i] = line
    return lines