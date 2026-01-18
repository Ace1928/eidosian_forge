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
def next_code():
    """Return integer of 'bitw' bits at 'bitcount' position in encoded."""
    start = bitcount // 8
    s = encoded[start:start + 4]
    try:
        code = unpack('>I', s)[0]
    except Exception:
        code = unpack('>I', s + b'\x00' * (4 - len(s)))[0]
    code <<= bitcount % 8
    code &= mask
    return code >> shr