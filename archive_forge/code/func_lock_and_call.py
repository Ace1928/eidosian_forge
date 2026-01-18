import functools
import threading
@functools.wraps(func)
@threadsafe_method
def lock_and_call(self, *args, **kwargs):
    with self:
        return func(self, *args, **kwargs)