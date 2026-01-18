from __future__ import print_function
import gc
import inspect
import os
import sys
import traceback
import types
from .debug import printExc
def getPreviousVersion(obj):
    """Return the previous version of *obj*, or None if this object has not
    been reloaded.
    """
    if isinstance(obj, type) or inspect.isfunction(obj):
        return getattr(obj, '__previous_reload_version__', None)
    elif inspect.ismethod(obj):
        if obj.__self__ is None:
            return getattr(obj.__func__, '__previous_reload_method__', None)
        else:
            oldmethod = getattr(obj.__func__, '__previous_reload_method__', None)
            if oldmethod is None:
                return None
            self = obj.__self__
            oldfunc = getattr(oldmethod, '__func__', oldmethod)
            if hasattr(oldmethod, 'im_class'):
                cls = oldmethod.im_class
                return types.MethodType(oldfunc, self, cls)
            else:
                return types.MethodType(oldfunc, self)