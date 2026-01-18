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
def byteorder_isnative(byteorder):
    """Return if byteorder matches the system's byteorder.

    >>> byteorder_isnative('=')
    True

    """
    if byteorder == '=' or byteorder == sys.byteorder:
        return True
    keys = {'big': '>', 'little': '<'}
    return keys.get(byteorder, byteorder) == keys[sys.byteorder]