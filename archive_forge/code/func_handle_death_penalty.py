import ctypes
import signal
import threading
def handle_death_penalty(self):
    """Raises an asynchronous exception in another thread.

        Reference http://docs.python.org/c-api/init.html#PyThreadState_SetAsyncExc for more info.
        """
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self._target_thread_id), ctypes.py_object(self._exception))
    if ret == 0:
        raise ValueError('Invalid thread ID {}'.format(self._target_thread_id))
    elif ret > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self._target_thread_id), 0)
        raise SystemError('PyThreadState_SetAsyncExc failed')