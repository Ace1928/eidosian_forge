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
def decode_jetraw(data: bytes | None, index: int, /, *, jpegtables: bytes | None=None, jpegheader: bytes | None=None, _fullsize: bool=False) -> tuple[NDArray[Any] | None, tuple[int, int, int, int, int], tuple[int, int, int, int]]:
    segmentindex, shape = indices(index)
    if data is None:
        if _fullsize:
            shape = pad_none(shape)
        return (data, segmentindex, shape)
    data_array = numpy.zeros(shape, numpy.uint16)
    decompress(data, out=data_array)
    return (data_array.reshape(shape), segmentindex, shape)