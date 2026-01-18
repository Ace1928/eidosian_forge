from weakref import ref
from contextlib import contextmanager
from threading import current_thread, RLock
def local_deleted(_, key=key):
    thread = wrthread()
    if thread is not None:
        del thread.__dict__[key]