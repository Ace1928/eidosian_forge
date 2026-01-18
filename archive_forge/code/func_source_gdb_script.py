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
def source_gdb_script(script_contents, to_string=False):
    """
    Source a gdb script with script_contents passed as a string. This is useful
    to provide defines for py-step and py-next to make them repeatable (this is
    not possible with gdb.execute()). See
    http://sourceware.org/bugzilla/show_bug.cgi?id=12216
    """
    fd, filename = tempfile.mkstemp()
    f = os.fdopen(fd, 'w')
    f.write(script_contents)
    f.close()
    gdb.execute('source %s' % filename, to_string=to_string)
    os.remove(filename)