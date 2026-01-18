from _pydev_bundle.pydev_imports import execfile
from _pydevd_bundle import pydevd_dont_trace
import types
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import get_global_debugger
def notify_info(*args):
    if DEBUG >= LEVEL1:
        write_err(*args)