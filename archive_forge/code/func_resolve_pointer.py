from __future__ import unicode_literals
from itertools import tee, chain
import re
import copy
def resolve_pointer(doc, pointer, default=_nothing):
    """ Resolves pointer against doc and returns the referenced object

    >>> obj = {'foo': {'anArray': [ {'prop': 44}], 'another prop': {'baz': 'A string' }}, 'a%20b': 1, 'c d': 2}

    >>> resolve_pointer(obj, '') == obj
    True

    >>> resolve_pointer(obj, '/foo') == obj['foo']
    True

    >>> resolve_pointer(obj, '/foo/another prop') == obj['foo']['another prop']
    True

    >>> resolve_pointer(obj, '/foo/another prop/baz') == obj['foo']['another prop']['baz']
    True

    >>> resolve_pointer(obj, '/foo/anArray/0') == obj['foo']['anArray'][0]
    True

    >>> resolve_pointer(obj, '/some/path', None) == None
    True

    >>> resolve_pointer(obj, '/a b', None) == None
    True

    >>> resolve_pointer(obj, '/a%20b') == 1
    True

    >>> resolve_pointer(obj, '/c d') == 2
    True

    >>> resolve_pointer(obj, '/c%20d', None) == None
    True
    """
    pointer = JsonPointer(pointer)
    return pointer.resolve(doc, default)