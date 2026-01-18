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
def jpeg_shape(jpeg: bytes, /) -> tuple[int, int, int, int]:
    """Return bitdepth and shape of JPEG image."""
    i = 0
    while i < len(jpeg):
        marker = struct.unpack('>H', jpeg[i:i + 2])[0]
        i += 2
        if marker == 65496:
            continue
        if marker == 65497:
            break
        if 65488 <= marker <= 65495:
            continue
        if marker == 65281:
            continue
        length = struct.unpack('>H', jpeg[i:i + 2])[0]
        i += 2
        if 65472 <= marker <= 65475:
            return struct.unpack('>BHHB', jpeg[i:i + 6])
        if marker == 65498:
            break
        i += length - 2
    raise ValueError('no SOF marker found')