from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def iterShapes(self, bbox=None):
    """Returns a generator of shapes in a shapefile. Useful
        for handling large shapefiles.
        To only read shapes within a given spatial region, specify the 'bbox'
        arg as a list or tuple of xmin,ymin,xmax,ymax. 
        """
    shp = self.__getFileObj(self.shp)
    shp.seek(0, 2)
    shpLength = shp.tell()
    shp.seek(100)
    if self.numShapes:
        for i in xrange(self.numShapes):
            shape = self.__shape(oid=i, bbox=bbox)
            if shape:
                yield shape
    else:
        i = 0
        offsets = []
        pos = shp.tell()
        while pos < shpLength:
            offsets.append(pos)
            shape = self.__shape(oid=i, bbox=bbox)
            pos = shp.tell()
            if shape:
                yield shape
            i += 1
        assert i == len(offsets)
        self.numShapes = i
        self._offsets = offsets