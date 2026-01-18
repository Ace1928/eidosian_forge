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
def read_colormap(fh, byteorder, dtype, count, offsetsize):
    """Read ColorMap data from file and return as numpy array."""
    cmap = fh.read_array(byteorder + dtype[-1], count)
    cmap.shape = (3, -1)
    return cmap