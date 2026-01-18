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
def read_sis_ini(fh: FileHandle, byteorder: ByteOrder, dtype: int, count: int, offsetsize: int, /) -> dict[str, Any]:
    """Read OlympusSIS INI string from file."""
    inistr = bytes2str(stripnull(fh.read(count)))
    try:
        return olympusini_metadata(inistr)
    except Exception as exc:
        logger().warning(f'<tifffile.olympusini_metadata> raised {exc!r}')
        return {}