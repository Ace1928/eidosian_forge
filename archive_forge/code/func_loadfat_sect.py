from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def loadfat_sect(self, sect):
    """
        Adds the indexes of the given sector to the FAT

        :param sect: string containing the first FAT sector, or array of long integers
        :returns: index of last FAT sector.
        """
    if isinstance(sect, array.array):
        fat1 = sect
    else:
        fat1 = self.sect2array(sect)
        if log.isEnabledFor(logging.DEBUG):
            self.dumpsect(sect)
    isect = None
    for isect in fat1:
        isect = isect & 4294967295
        log.debug('isect = %X' % isect)
        if isect == ENDOFCHAIN or isect == FREESECT:
            log.debug('found end of sector chain')
            break
        s = self.getsect(isect)
        nextfat = self.sect2array(s)
        self.fat = self.fat + nextfat
    return isect