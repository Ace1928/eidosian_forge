from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def filter_output(regexes):
    output = []
    for regex, flags in regexes:
        for match in re.finditer(regex, result, flags):
            output.append(match.group(0))
    return '\n'.join(output)