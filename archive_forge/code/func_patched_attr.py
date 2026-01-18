from _pydev_bundle._pydev_saved_modules import threading
def patched_attr(*args, **kwargs):
    self.call_begin(attr)
    result = orig_attr(*args, **kwargs)
    self.call_end(attr)
    if result == self.wrapped_object:
        return self
    return result