from contextlib import contextmanager
import os
import shutil
import tempfile
import struct
def partition_all(n, bytes):
    """ Partition bytes into evenly sized blocks

    The final block holds the remainder and so may not be of equal size

    >>> list(partition_all(2, b'Hello'))
    [b'He', b'll', b'o']

    See Also:
        toolz.partition_all
    """
    if len(bytes) < n:
        yield bytes
    else:
        for i in range(0, len(bytes), n):
            yield bytes[i:i + n]