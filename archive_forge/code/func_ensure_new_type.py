import types
import sys
import numbers
import functools
import copy
import inspect
def ensure_new_type(obj):
    from future.types.newbytes import newbytes
    from future.types.newstr import newstr
    from future.types.newint import newint
    from future.types.newdict import newdict
    native_type = type(native(obj))
    if issubclass(native_type, type(obj)):
        if native_type == str:
            return newbytes(obj)
        elif native_type == unicode:
            return newstr(obj)
        elif native_type == int:
            return newint(obj)
        elif native_type == long:
            return newint(obj)
        elif native_type == dict:
            return newdict(obj)
        else:
            return obj
    else:
        assert type(obj) in [newbytes, newstr]
        return obj