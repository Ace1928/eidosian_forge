import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
def get_after_tokens(self):
    ret = self._after_tokens.copy()
    for handler in self._after_handler_tokens:
        ret.update(handler.tokens)
    return ret