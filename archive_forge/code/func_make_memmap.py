import os
import re
import time
from os.path import basename
from multiprocessing import util
def make_memmap(filename, dtype='uint8', mode='r+', offset=0, shape=None, order='C', unlink_on_gc_collect=False):
    raise NotImplementedError("'joblib.backports.make_memmap' should not be used if numpy is not installed.")