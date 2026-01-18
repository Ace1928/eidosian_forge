import io
import os
import sys
from pyasn1 import error
from pyasn1.type import univ
def peekIntoStream(substrate, size=-1):
    """Peek into stream.

    Parameters
    ----------
    substrate: :py:class:`IOBase`
        Stream to read from.

    size: :py:class:`int`
        How many bytes to peek (-1 = all available)

    Returns
    -------
    : :py:class:`bytes` or :py:class:`str`
        The return type depends on Python major version
    """
    if hasattr(substrate, 'peek'):
        received = substrate.peek(size)
        if received is None:
            yield
        while len(received) < size:
            yield
        yield received
    else:
        current_position = substrate.tell()
        try:
            for chunk in readFromStream(substrate, size):
                yield chunk
        finally:
            substrate.seek(current_position)