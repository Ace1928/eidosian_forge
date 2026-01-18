from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def loadminifat(self):
    """
        Load the MiniFAT table.
        """
    stream_size = self.num_mini_fat_sectors * self.sector_size
    nb_minisectors = (self.root.size + self.mini_sector_size - 1) // self.mini_sector_size
    used_size = nb_minisectors * 4
    log.debug('loadminifat(): minifatsect=%d, nb FAT sectors=%d, used_size=%d, stream_size=%d, nb MiniSectors=%d' % (self.minifatsect, self.num_mini_fat_sectors, used_size, stream_size, nb_minisectors))
    if used_size > stream_size:
        self._raise_defect(DEFECT_INCORRECT, 'OLE MiniStream is larger than MiniFAT')
    s = self._open(self.minifatsect, stream_size, force_FAT=True).read()
    self.minifat = self.sect2array(s)
    log.debug('MiniFAT shrunk from %d to %d sectors' % (len(self.minifat), nb_minisectors))
    self.minifat = self.minifat[:nb_minisectors]
    log.debug('loadminifat(): len=%d' % len(self.minifat))
    if log.isEnabledFor(logging.DEBUG):
        log.debug('\nMiniFAT:')
        self.dumpfat(self.minifat)