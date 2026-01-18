import shutil
import sys
import tempfile
import unittest
import httplib2
from lazr.restfulclient._browser import AtomicFileCache, safename
@staticmethod
def prefix_safename(x):
    if isinstance(x, binary_type):
        x = x.decode('utf-8')
    return AtomicFileCache.TEMPFILE_PREFIX + x