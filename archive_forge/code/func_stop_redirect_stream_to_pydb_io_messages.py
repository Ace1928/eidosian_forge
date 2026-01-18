from _pydevd_bundle.pydevd_constants import ForkSafeLock, get_global_debugger
import os
import sys
from contextlib import contextmanager
def stop_redirect_stream_to_pydb_io_messages(std):
    """
    :param std:
        'stdout' or 'stderr'
    """
    with _RedirectionsHolder._lock:
        redirect_to_name = '_pydevd_%s_redirect_' % (std,)
        redirect_info = getattr(_RedirectionsHolder, redirect_to_name)
        if redirect_info is not None:
            setattr(_RedirectionsHolder, redirect_to_name, None)
            stack = getattr(_RedirectionsHolder, '_stack_%s' % std)
            prev_info = stack.pop()
            curr = getattr(sys, std)
            if curr is redirect_info.redirect_to:
                setattr(sys, std, redirect_info.original)