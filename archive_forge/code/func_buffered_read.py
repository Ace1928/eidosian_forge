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
def buffered_read(fh, lock, offsets, bytecounts, buffersize=2 ** 26):
    """Return iterator over blocks read from file."""
    length = len(offsets)
    i = 0
    while i < length:
        data = []
        with lock:
            size = 0
            while size < buffersize and i < length:
                fh.seek(offsets[i])
                bytecount = bytecounts[i]
                data.append(fh.read(bytecount))
                size += bytecount
                i += 1
        for block in data:
            yield block