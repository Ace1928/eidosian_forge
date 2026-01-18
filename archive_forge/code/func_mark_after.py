import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
def mark_after(self, v):
    if isinstance(v, _Token):
        self._after_tokens.add(v)
    elif isinstance(v, _BaseHandler):
        self._after_handler_tokens.add(v)
    else:
        raise AssertionError('Unhandled: %s' % (v,))