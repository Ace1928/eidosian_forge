from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def dumpfat(self, fat, firstindex=0):
    """
        Display a part of FAT in human-readable form for debugging purposes
        """
    VPL = 8
    fatnames = {FREESECT: '..free..', ENDOFCHAIN: '[ END. ]', FATSECT: 'FATSECT ', DIFSECT: 'DIFSECT '}
    nbsect = len(fat)
    nlines = (nbsect + VPL - 1) // VPL
    print('index', end=' ')
    for i in range(VPL):
        print('%8X' % i, end=' ')
    print()
    for l in range(nlines):
        index = l * VPL
        print('%6X:' % (firstindex + index), end=' ')
        for i in range(index, index + VPL):
            if i >= nbsect:
                break
            sect = fat[i]
            aux = sect & 4294967295
            if aux in fatnames:
                name = fatnames[aux]
            elif sect == i + 1:
                name = '    --->'
            else:
                name = '%8X' % sect
            print(name, end=' ')
        print()