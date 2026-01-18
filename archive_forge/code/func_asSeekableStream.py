import io
import os
import sys
from pyasn1 import error
from pyasn1.type import univ
def asSeekableStream(substrate):
    """Convert object to seekable byte-stream.

    Parameters
    ----------
    substrate: :py:class:`bytes` or :py:class:`io.IOBase` or :py:class:`univ.OctetString`

    Returns
    -------
    : :py:class:`io.IOBase`

    Raises
    ------
    : :py:class:`~pyasn1.error.PyAsn1Error`
        If the supplied substrate cannot be converted to a seekable stream.
    """
    if isinstance(substrate, io.BytesIO):
        return substrate
    elif isinstance(substrate, bytes):
        return io.BytesIO(substrate)
    elif isinstance(substrate, univ.OctetString):
        return io.BytesIO(substrate.asOctets())
    try:
        if _PY2 and isinstance(substrate, file):
            return io.BufferedReader(substrate)
        elif substrate.seekable():
            return substrate
        else:
            return CachingStreamWrapper(substrate)
    except AttributeError:
        raise error.UnsupportedSubstrateError('Cannot convert ' + substrate.__class__.__name__ + ' to a seekable bit stream.')