from _pydevd_bundle.pydevd_constants import ForkSafeLock, get_global_debugger
import os
import sys
from contextlib import contextmanager
@contextmanager
def redirect_stream_to_pydb_io_messages_context():
    with _RedirectionsHolder._lock:
        redirecting = []
        for std in ('stdout', 'stderr'):
            if redirect_stream_to_pydb_io_messages(std):
                redirecting.append(std)
        try:
            yield
        finally:
            for std in redirecting:
                stop_redirect_stream_to_pydb_io_messages(std)