from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def getclsid(self, filename):
    """
        Return clsid of a stream/storage.

        :param filename: path of stream/storage in storage tree. (see openstream for
            syntax)
        :returns: Empty string if clsid is null, a printable representation of the clsid otherwise

        new in version 0.44
        """
    sid = self._find(filename)
    entry = self.direntries[sid]
    return entry.clsid