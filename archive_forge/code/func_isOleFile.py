from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def isOleFile(filename=None, data=None):
    """
    Test if a file is an OLE container (according to the magic bytes in its header).

    .. note::
        This function only checks the first 8 bytes of the file, not the
        rest of the OLE structure.
        If data is provided, it also checks if the file size is above
        the minimal size of an OLE file (1536 bytes).
        If filename is provided with the path of the file on disk, the file is
        open only to read the first 8 bytes, then closed.

    .. versionadded:: 0.16

    :param filename: filename, contents or file-like object of the OLE file (string-like or file-like object)

        - if data is provided, filename is ignored.
        - if filename is a unicode string, it is used as path of the file to open on disk.
        - if filename is a bytes string smaller than 1536 bytes, it is used as path
          of the file to open on disk.
        - [deprecated] if filename is a bytes string longer than 1535 bytes, it is parsed
          as the content of an OLE file in memory. (bytes type only)
          Note that this use case is deprecated and should be replaced by the new data parameter
        - if filename is a file-like object (with read and seek methods),
          it is parsed as-is.
    :type filename: bytes, str, unicode or file-like object

    :param data: bytes string with the contents of the file to be checked, when the file is in memory
                 (added in olefile 0.47)
    :type data: bytes

    :returns: True if OLE, False otherwise.
    :rtype: bool
    """
    header = None
    if data is not None:
        if len(data) >= MINIMAL_OLEFILE_SIZE:
            header = data[:len(MAGIC)]
        else:
            return False
    elif hasattr(filename, 'read') and hasattr(filename, 'seek'):
        header = filename.read(len(MAGIC))
        filename.seek(0)
    elif isinstance(filename, bytes) and len(filename) >= MINIMAL_OLEFILE_SIZE:
        header = filename[:len(MAGIC)]
    else:
        with open(filename, 'rb') as fp:
            header = fp.read(len(MAGIC))
    if header == MAGIC:
        return True
    else:
        return False