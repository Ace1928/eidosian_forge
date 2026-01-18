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
def iterRecords(self, fields=None):
    """Returns a generator of records in a dbf file.
        Useful for large shapefiles or dbf files.
        To only read some of the fields, specify the 'fields' arg as a
        list of one or more fieldnames.
        """
    if self.numRecords is None:
        self.__dbfHeader()
    f = self.__getFileObj(self.dbf)
    f.seek(self.__dbfHdrLength)
    fieldTuples, recLookup, recStruct = self.__recordFields(fields)
    for i in xrange(self.numRecords):
        r = self.__record(oid=i, fieldTuples=fieldTuples, recLookup=recLookup, recStruct=recStruct)
        if r:
            yield r