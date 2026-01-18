import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def printdir(self, file=None):
    """Print a table of contents for the zip file."""
    print('%-46s %19s %12s' % ('File Name', 'Modified    ', 'Size'), file=file)
    for zinfo in self.filelist:
        date = '%d-%02d-%02d %02d:%02d:%02d' % zinfo.date_time[:6]
        print('%-46s %s %12d' % (zinfo.filename, date, zinfo.file_size), file=file)