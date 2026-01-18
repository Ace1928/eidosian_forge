import io
import os
import sys
from pyasn1 import error
from pyasn1.type import univ
def readFromStream(substrate, size=-1, context=None):
    """Read from the stream.

    Parameters
    ----------
    substrate: :py:class:`IOBase`
        Stream to read from.

    Keyword parameters
    ------------------
    size: :py:class:`int`
        How many bytes to read (-1 = all available)

    context: :py:class:`dict`
        Opaque caller context will be attached to exception objects created
        by this function.

    Yields
    ------
    : :py:class:`bytes` or :py:class:`str` or :py:class:`SubstrateUnderrunError`
        Read data or :py:class:`~pyasn1.error.SubstrateUnderrunError`
        object if no `size` bytes is readily available in the stream. The
        data type depends on Python major version

    Raises
    ------
    : :py:class:`~pyasn1.error.EndOfStreamError`
        Input stream is exhausted
    """
    while True:
        received = substrate.read(size)
        if received is None:
            yield error.SubstrateUnderrunError(context=context)
        elif not received and size != 0:
            raise error.EndOfStreamError(context=context)
        elif len(received) < size:
            substrate.seek(-len(received), os.SEEK_CUR)
            yield error.SubstrateUnderrunError(context=context)
        else:
            break
    yield received