from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def loadfat(self, header):
    """
        Load the FAT table.
        """
    log.debug('Loading the FAT table, starting with the 1st sector after the header')
    sect = header[76:512]
    log.debug('len(sect)=%d, so %d integers' % (len(sect), len(sect) // 4))
    self.fat = array.array(UINT32)
    self.loadfat_sect(sect)
    if self.num_difat_sectors != 0:
        log.debug('DIFAT is used, because file size > 6.8MB.')
        if self.num_fat_sectors <= 109:
            self._raise_defect(DEFECT_INCORRECT, 'incorrect DIFAT, not enough sectors')
        if self.first_difat_sector >= self.nb_sect:
            self._raise_defect(DEFECT_FATAL, 'incorrect DIFAT, first index out of range')
        log.debug('DIFAT analysis...')
        nb_difat_sectors = self.sectorsize // 4 - 1
        nb_difat = (self.num_fat_sectors - 109 + nb_difat_sectors - 1) // nb_difat_sectors
        log.debug('nb_difat = %d' % nb_difat)
        if self.num_difat_sectors != nb_difat:
            raise IOError('incorrect DIFAT')
        isect_difat = self.first_difat_sector
        for i in iterrange(nb_difat):
            log.debug('DIFAT block %d, sector %X' % (i, isect_difat))
            sector_difat = self.getsect(isect_difat)
            difat = self.sect2array(sector_difat)
            if log.isEnabledFor(logging.DEBUG):
                self.dumpsect(sector_difat)
            self.loadfat_sect(difat[:nb_difat_sectors])
            isect_difat = difat[nb_difat_sectors]
            log.debug('next DIFAT sector: %X' % isect_difat)
        if isect_difat not in [ENDOFCHAIN, FREESECT]:
            raise IOError('incorrect end of DIFAT')
    else:
        log.debug('No DIFAT, because file size < 6.8MB.')
    if len(self.fat) > self.nb_sect:
        log.debug('len(fat)=%d, shrunk to nb_sect=%d' % (len(self.fat), self.nb_sect))
        self.fat = self.fat[:self.nb_sect]
    log.debug('FAT references %d sectors / Maximum %d sectors in file' % (len(self.fat), self.nb_sect))
    if log.isEnabledFor(logging.DEBUG):
        log.debug('\nFAT:')
        self.dumpfat(self.fat)