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
def ndpi_jpeg_tile(jpeg: bytes, /) -> tuple[int, int, bytes]:
    """Return tile shape and JPEG header from JPEG with restart markers."""
    marker: int
    length: int
    factor: int
    ncomponents: int
    restartinterval: int = 0
    sofoffset: int = 0
    sosoffset: int = 0
    i: int = 0
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
        if marker == 65501:
            restartinterval = struct.unpack('>H', jpeg[i:i + 2])[0]
        elif marker == 65472:
            sofoffset = i + 1
            precision, imlength, imwidth, ncomponents = struct.unpack('>BHHB', jpeg[i:i + 6])
            i += 6
            mcuwidth = 1
            mcuheight = 1
            for _ in range(ncomponents):
                cid, factor, table = struct.unpack('>BBB', jpeg[i:i + 3])
                i += 3
                if factor >> 4 > mcuwidth:
                    mcuwidth = factor >> 4
                if factor & 15 > mcuheight:
                    mcuheight = factor & 15
            mcuwidth *= 8
            mcuheight *= 8
            i = sofoffset - 1
        elif marker == 65498:
            sosoffset = i + length - 2
            break
        i += length - 2
    if restartinterval == 0 or sofoffset == 0 or sosoffset == 0:
        raise ValueError('missing required JPEG markers')
    tilelength = mcuheight
    tilewidth = restartinterval * mcuwidth
    jpegheader = jpeg[:sofoffset] + struct.pack('>HH', tilelength, tilewidth) + jpeg[sofoffset + 4:sosoffset]
    return (tilelength, tilewidth, jpegheader)