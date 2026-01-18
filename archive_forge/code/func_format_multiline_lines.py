import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
def format_multiline_lines(lines):
    """Same as format_multline, but taking input pre-split into lines."""
    out_lines = []
    for i, line in enumerate(lines):
        if i != 0:
            if not line.strip():
                line = '.'
            line = ' ' + line
        out_lines.append(line)
    return '\n'.join(out_lines)