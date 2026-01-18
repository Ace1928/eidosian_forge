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
def stripnull(string, null=b'\x00'):
    """Return string truncated at first null character.

    Clean NULL terminated C strings. For unicode strings use null='\\0'.

    >>> stripnull(b'string\\x00')
    b'string'
    >>> stripnull('string\\x00', null='\\0')
    'string'

    """
    i = string.find(null)
    return string if i < 0 else string[:i]