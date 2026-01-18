import functools
import threading
@staticmethod
def assert_locked(self):
    lock = type(self)._lock
    assert lock.acquire(blocking=False), 'ThreadSafeSingleton accessed without locking. Either use with-statement, or if it is a method or property, mark it as @threadsafe_method or with @autolocked_method, as appropriate.'
    lock.release()