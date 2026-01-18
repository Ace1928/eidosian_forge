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
@cached_property
def ndpi_tags(self) -> dict[str, Any] | None:
    """Consolidated metadata from Hamamatsu NDPI tags."""
    if not self.is_ndpi:
        return None
    tags = self.tags
    result = {}
    for name in ('Make', 'Model', 'Software'):
        result[name] = tags[name].value
    for code, name in TIFF.NDPI_TAGS.items():
        if code in tags:
            result[name] = tags[code].value
    if 'McuStarts' in result:
        mcustarts = result['McuStarts']
        if 'McuStartsHighBytes' in result:
            high = result['McuStartsHighBytes'].astype('uint64')
            high <<= 32
            mcustarts = mcustarts.astype('uint64')
            mcustarts += high
            del result['McuStartsHighBytes']
        result['McuStarts'] = mcustarts
    return result