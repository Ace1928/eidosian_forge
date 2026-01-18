from __future__ import absolute_import
import sys
from contextlib import contextmanager
from ..Utils import open_new_file
from . import DebugFlags
from . import Options
def performance_hint(position, message, env):
    if not env.directives['show_performance_hints']:
        return
    warn = CompileWarning(position, message)
    line = 'performance hint: %s\n' % warn
    listing_file = threadlocal.cython_errors_listing_file
    if listing_file:
        _write_file_encode(listing_file, line)
    echo_file = threadlocal.cython_errors_echo_file
    if echo_file:
        _write_file_encode(echo_file, line)
    return warn