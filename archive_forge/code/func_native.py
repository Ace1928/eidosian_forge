import types
import sys
import numbers
import functools
import copy
import inspect
def native(obj):
    """
    On Py3, this is a no-op: native(obj) -> obj

    On Py2, returns the corresponding native Py2 types that are
    superclasses for backported objects from Py3:

    >>> from builtins import str, bytes, int

    >>> native(str(u'ABC'))
    u'ABC'
    >>> type(native(str(u'ABC')))
    unicode

    >>> native(bytes(b'ABC'))
    b'ABC'
    >>> type(native(bytes(b'ABC')))
    bytes

    >>> native(int(10**20))
    100000000000000000000L
    >>> type(native(int(10**20)))
    long

    Existing native types on Py2 will be returned unchanged:

    >>> type(native(u'ABC'))
    unicode
    """
    if hasattr(obj, '__native__'):
        return obj.__native__()
    else:
        return obj