import atexit
import queue
import threading
import weakref
@classmethod
def create_and_register(cls, executor, work_queue):
    w = cls(executor, work_queue)
    _to_be_cleaned[w] = True
    return w