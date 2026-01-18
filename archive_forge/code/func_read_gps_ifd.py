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
def read_gps_ifd(fh, byteorder, dtype, count, offsetsize):
    """Read GPS tags from file and return as dict."""
    return read_tags(fh, byteorder, offsetsize, TIFF.GPS_TAGS, maxifds=1)