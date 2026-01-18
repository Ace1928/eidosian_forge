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
def load_shx(self, shapefile_name):
    """
        Attempts to load file with .shx extension as both lower and upper case
        """
    shx_ext = 'shx'
    try:
        self.shx = open('%s.%s' % (shapefile_name, shx_ext), 'rb')
        self._files_to_close.append(self.shx)
    except IOError:
        try:
            self.shx = open('%s.%s' % (shapefile_name, shx_ext.upper()), 'rb')
            self._files_to_close.append(self.shx)
        except IOError:
            pass