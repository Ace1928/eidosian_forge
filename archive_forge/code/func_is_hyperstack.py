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
def is_hyperstack():
    if not page.is_final:
        return False
    images = ij.get('images', 0)
    if images <= 1:
        return False
    offset, count = page.is_contiguous
    if count != product(page.shape) * page.bitspersample // 8 or offset + count * images > self.filehandle.size:
        raise ValueError()
    if len(pages) > 1 and offset + count * images > pages[1].offset:
        return False
    return True