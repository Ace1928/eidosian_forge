import atexit
import queue
import threading
import weakref
@staticmethod
def lock_object(*args, **kwargs):
    return threading.Lock(*args, **kwargs)