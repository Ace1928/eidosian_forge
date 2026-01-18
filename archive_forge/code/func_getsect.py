from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def getsect(self, sect):
    """
        Read given sector from file on disk.

        :param sect: int, sector index
        :returns: a string containing the sector data.
        """
    try:
        self.fp.seek(self.sectorsize * (sect + 1))
    except Exception:
        log.debug('getsect(): sect=%X, seek=%d, filesize=%d' % (sect, self.sectorsize * (sect + 1), self._filesize))
        self._raise_defect(DEFECT_FATAL, 'OLE sector index out of range')
    sector = self.fp.read(self.sectorsize)
    if len(sector) != self.sectorsize:
        log.debug('getsect(): sect=%X, read=%d, sectorsize=%d' % (sect, len(sector), self.sectorsize))
        self._raise_defect(DEFECT_FATAL, 'incomplete OLE sector')
    return sector