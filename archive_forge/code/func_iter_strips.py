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
def iter_strips(pageiter: Iterator[NDArray[Any] | None], shape: tuple[int, ...], dtype: numpy.dtype[Any], rowsperstrip: int, /) -> Iterator[NDArray[Any]]:
    """Return iterator over strips in pages."""
    numstrips = (shape[-3] + rowsperstrip - 1) // rowsperstrip
    for iteritem in pageiter:
        if iteritem is None:
            pagedata = numpy.zeros(shape, dtype)
        else:
            pagedata = iteritem.reshape(shape)
        for plane in pagedata:
            for depth in plane:
                for i in range(numstrips):
                    yield depth[i * rowsperstrip:(i + 1) * rowsperstrip]