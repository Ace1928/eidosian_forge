from _pydevd_bundle.pydevd_constants import get_frame, IS_CPYTHON, IS_64BIT_PROCESS, IS_WINDOWS, \
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import pydev_log, pydev_monkey
import os.path
import platform
import ctypes
from io import StringIO
import sys
import traceback
def set_trace_to_threads(tracing_func, thread_idents=None, create_dummy_thread=True):
    assert tracing_func is not None
    ret = 0
    if thread_idents is None:
        thread_idents = set(sys._current_frames().keys())
        for t in threading.enumerate():
            if getattr(t, 'pydev_do_not_trace', False):
                thread_idents.discard(t.ident)
            else:
                thread_idents.add(t.ident)
    curr_ident = thread.get_ident()
    curr_thread = threading._active.get(curr_ident)
    if curr_ident in thread_idents and len(thread_idents) != 1:
        thread_idents = list(thread_idents)
        thread_idents.remove(curr_ident)
        thread_idents.insert(0, curr_ident)
    for thread_ident in thread_idents:
        if create_dummy_thread:
            if thread_ident not in threading._active:

                class _DummyThread(threading._DummyThread):

                    def _set_ident(self):
                        self._ident = thread_ident
                t = _DummyThread()
                t.__class__ = threading._DummyThread
                if thread_ident == curr_ident:
                    curr_thread = t
                with threading._active_limbo_lock:
                    threading._active[thread_ident] = t
                    threading._active[curr_ident] = curr_thread
                    if t.ident != thread_ident:
                        pydev_log.critical('pydevd: creation of _DummyThread with fixed thread ident did not succeed.')
        show_debug_info = 0
        proceed = thread.allocate_lock()
        proceed.acquire()

        def dummy_trace(frame, event, arg):
            return dummy_trace

        def increase_tracing_count():
            set_trace = TracingFunctionHolder._original_tracing or sys.settrace
            set_trace(dummy_trace)
            proceed.release()
        start_new_thread = pydev_monkey.get_original_start_new_thread(thread)
        start_new_thread(increase_tracing_count, ())
        proceed.acquire()
        proceed = None
        set_trace_func = TracingFunctionHolder._original_tracing or sys.settrace
        lib = _load_python_helper_lib()
        if lib is None:
            pydev_log.info('Unable to load helper lib to set tracing to all threads (unsupported python vm).')
            ret = -1
        else:
            try:
                result = lib.AttachDebuggerTracing(ctypes.c_int(show_debug_info), ctypes.py_object(set_trace_func), ctypes.py_object(tracing_func), ctypes.c_uint(thread_ident), ctypes.py_object(None))
            except:
                if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
                    pydev_log.exception('Error attaching debugger tracing')
                ret = -1
            else:
                if result != 0:
                    pydev_log.info('Unable to set tracing for existing thread. Result: %s', result)
                    ret = result
    return ret