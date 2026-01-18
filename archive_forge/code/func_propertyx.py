import functools
import warnings
import threading
import sys
def propertyx(function):
    """Decorator to easily create properties in classes.

    Example:

        class Angle(object):
            def __init__(self, rad):
                self._rad = rad

            @property
            def rad():
                def fget(self):
                    return self._rad
                def fset(self, angle):
                    if isinstance(angle, Angle):
                        angle = angle.rad
                    self._rad = float(angle)

    Arguments:
    - `function`: The function to be decorated.
    """
    keys = ('fget', 'fset', 'fdel')
    func_locals = {'doc': function.__doc__}

    def probe_func(frame, event, arg):
        if event == 'return':
            locals = frame.f_locals
            func_locals.update(dict(((k, locals.get(k)) for k in keys)))
            sys.settrace(None)
        return probe_func
    sys.settrace(probe_func)
    function()
    return property(**func_locals)