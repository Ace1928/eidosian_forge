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
def os_fsencode(filename):
    if not isinstance(filename, unicode):
        return filename
    encoding = sys.getfilesystemencoding()
    if encoding == 'mbcs':
        return filename.encode(encoding)
    encoded = []
    for char in filename:
        if 56448 <= ord(char) <= 56575:
            byte = chr(ord(char) - 56320)
        else:
            byte = char.encode(encoding)
        encoded.append(byte)
    return ''.join(encoded)