from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def read_lsm_channeldatatypes(fh: FileHandle, /) -> NDArray[Any]:
    """Read LSM channel data type from file."""
    size = struct.unpack('<I', fh.read(4))[0]
    return fh.read_array('<u4', count=size)