from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def dumpsect(self, sector, firstindex=0):
    """
        Display a sector in a human-readable form, for debugging purposes
        """
    VPL = 8
    tab = array.array(UINT32, sector)
    if sys.byteorder == 'big':
        tab.byteswap()
    nbsect = len(tab)
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
            sect = tab[i]
            name = '%8X' % sect
            print(name, end=' ')
        print()