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
def load_dbf(self, shapefile_name):
    """
        Attempts to load file with .dbf extension as both lower and upper case
        """
    dbf_ext = 'dbf'
    try:
        self.dbf = open('%s.%s' % (shapefile_name, dbf_ext), 'rb')
        self._files_to_close.append(self.dbf)
    except IOError:
        try:
            self.dbf = open('%s.%s' % (shapefile_name, dbf_ext.upper()), 'rb')
            self._files_to_close.append(self.dbf)
        except IOError:
            pass