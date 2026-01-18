import sys
from winappdbg import win32
from winappdbg.system import System
from winappdbg.process import Process
from winappdbg.thread import Thread
from winappdbg.module import Module
from winappdbg.window import Window
from winappdbg.breakpoint import _BreakpointContainer, CodeBreakpoint
from winappdbg.event import Event, EventHandler, EventDispatcher, EventFactory
from winappdbg.interactive import ConsoleDebugger
import warnings
@staticmethod
def force_garbage_collection(bIgnoreExceptions=True):
    """
        Close all Win32 handles the Python garbage collector failed to close.

        @type  bIgnoreExceptions: bool
        @param bIgnoreExceptions: C{True} to ignore any exceptions that may be
            raised when detaching.
        """
    try:
        import gc
        gc.collect()
        bRecollect = False
        for obj in list(gc.garbage):
            try:
                if isinstance(obj, win32.Handle):
                    obj.close()
                elif isinstance(obj, Event):
                    obj.debug = None
                elif isinstance(obj, Process):
                    obj.clear()
                elif isinstance(obj, Thread):
                    obj.set_process(None)
                    obj.clear()
                elif isinstance(obj, Module):
                    obj.set_process(None)
                elif isinstance(obj, Window):
                    obj.set_process(None)
                else:
                    continue
                gc.garbage.remove(obj)
                del obj
                bRecollect = True
            except Exception:
                if not bIgnoreExceptions:
                    raise
                e = sys.exc_info()[1]
                warnings.warn(str(e), RuntimeWarning)
        if bRecollect:
            gc.collect()
    except Exception:
        if not bIgnoreExceptions:
            raise
        e = sys.exc_info()[1]
        warnings.warn(str(e), RuntimeWarning)