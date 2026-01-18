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
def subresolution(a: TiffPage | TiffPageSeries, b: TiffPage | TiffPageSeries, /, p: int=2, n: int=16) -> int | None:
    """Return level of subresolution of series or page b vs a."""
    if a.axes != b.axes or a.dtype != b.dtype:
        return None
    level = None
    for ax, i, j in zip(a.axes.lower(), a.shape, b.shape):
        if ax in 'xyz':
            if level is None:
                for r in range(n):
                    d = p ** r
                    if d > i:
                        return None
                    if abs(i / d - j) < 1.0:
                        level = r
                        break
                else:
                    return None
            else:
                d = p ** level
                if d > i:
                    return None
                if abs(i / d - j) >= 1.0:
                    return None
        elif i != j:
            return None
    return level