from _pydev_bundle._pydev_saved_modules import threading
def factory_wrapper(fun):

    def inner(*args, **kwargs):
        obj = fun(*args, **kwargs)
        return ObjectWrapper(obj)
    return inner