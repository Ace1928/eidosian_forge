from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def memmap_array(self, dtype, shape, offset=0, mode='r', order='C'):
    """Return numpy.memmap of data stored in file."""
    if not self.is_file:
        raise ValueError('Cannot memory-map file without fileno')
    return numpy.memmap(self._fh, dtype=dtype, mode=mode, offset=self._offset + offset, shape=shape, order=order)