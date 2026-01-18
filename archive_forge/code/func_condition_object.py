import atexit
import queue
import threading
import weakref
@staticmethod
def condition_object(*args, **kwargs):
    return threading.Condition(*args, **kwargs)