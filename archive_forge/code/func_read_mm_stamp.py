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
def read_mm_stamp(fh, byteorder, dtype, count, offsetsize):
    """Read FluoView mm_stamp tag from file and return as numpy.ndarray."""
    return fh.read_array(byteorder + 'f8', 8)