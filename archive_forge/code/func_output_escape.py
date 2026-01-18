import functools
import re
import sys
from Xlib.support import lock
def output_escape(value):
    value = str(value)
    if not value:
        return value
    for char, esc in (('\\', '\\\\'), ('\x00', '\\000'), ('\n', '\\n')):
        value = value.replace(char, esc)
    if value[0] in ' \t':
        value = '\\' + value
    if value[-1] in ' \t' and value[-2:-1] != '\\':
        value = value[:-1] + '\\' + value[-1]
    return value