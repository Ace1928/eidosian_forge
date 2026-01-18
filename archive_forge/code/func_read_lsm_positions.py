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
def read_lsm_positions(fh):
    """Read LSM positions from file and return as list."""
    size = struct.unpack('<I', fh.read(4))[0]
    return fh.read_array('<2f8', count=size)