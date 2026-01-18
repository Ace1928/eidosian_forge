import os
import mimetypes
from typing import Generator
from libcloud.utils.py3 import b, next
def exhaust_iterator(iterator):
    """
    Exhaust an iterator and return all data returned by it.

    :type iterator: :class:`object` which implements iterator interface.
    :param iterator: An object which implements an iterator interface
                     or a File like object with read method.

    :rtype ``str``
    :return Data returned by the iterator.
    """
    data = b('')
    try:
        chunk = b(next(iterator))
    except StopIteration:
        chunk = b('')
    while len(chunk) > 0:
        data += chunk
        try:
            chunk = b(next(iterator))
        except StopIteration:
            chunk = b('')
    return data