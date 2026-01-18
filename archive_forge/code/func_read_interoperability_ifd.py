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
def read_interoperability_ifd(fh, byteorder, dtype, count, offsetsize):
    """Read Interoperability tags from file and return as dict."""
    tag_names = {1: 'InteroperabilityIndex'}
    return read_tags(fh, byteorder, offsetsize, tag_names, maxifds=1)