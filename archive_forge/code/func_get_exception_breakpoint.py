from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_import_class
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame
from _pydev_bundle._pydev_saved_modules import threading
def get_exception_breakpoint(exctype, exceptions):
    if not exctype:
        exception_full_qname = None
    else:
        exception_full_qname = str(exctype.__module__) + '.' + exctype.__name__
    exc = None
    if exceptions is not None:
        try:
            return exceptions[exception_full_qname]
        except KeyError:
            for exception_breakpoint in exceptions.values():
                if exception_breakpoint.type is not None and issubclass(exctype, exception_breakpoint.type):
                    if exc is None or issubclass(exception_breakpoint.type, exc.type):
                        exc = exception_breakpoint
    return exc