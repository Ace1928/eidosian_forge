from __future__ import (absolute_import, division, print_function)
from re import compile as re_compile
from ansible.plugins.become import BecomeBase
from ansible.module_utils._text import to_bytes
@staticmethod
def remove_ansi_codes(line):
    return ansi_color_codes.sub(b'', line)