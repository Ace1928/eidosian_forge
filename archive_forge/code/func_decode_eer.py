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
def decode_eer(data: bytes | None, index: int, /, *, jpegtables: bytes | None=None, jpegheader: bytes | None=None, _fullsize: bool=False) -> tuple[NDArray[Any] | None, tuple[int, int, int, int, int], tuple[int, int, int, int]]:
    segmentindex, shape = indices(index)
    if data is None:
        if _fullsize:
            shape = pad_none(shape)
        return (data, segmentindex, shape)
    data_array = decompress(data, shape=shape[1:3], rlebits=rlebits, horzbits=horzbits, vertbits=vertbits, superres=False)
    return (data_array.reshape(shape), segmentindex, shape)