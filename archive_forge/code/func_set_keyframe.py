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
def set_keyframe(self, index: int, /) -> None:
    """Set keyframe to TiffPage specified by `index`.

        If not found in the cache, the TiffPage at `index` is loaded from file
        and added to the cache.

        """
    if not isinstance(index, (int, numpy.integer)):
        raise TypeError(f'indices must be integers, not {type(index)}')
    index = int(index)
    if index < 0:
        index %= len(self)
    if self._keyframe is not None and self._keyframe.index == index:
        return
    if index == 0:
        self._keyframe = cast(TiffPage, self.pages[0])
        return
    if self._indexed or index < len(self.pages):
        page = self.pages[index]
        if isinstance(page, TiffPage):
            self._keyframe = page
            return
        if isinstance(page, TiffFrame):
            self.pages[index] = page.offset
    tiffpage = self._tiffpage
    self._tiffpage = TiffPage
    try:
        self._keyframe = cast(TiffPage, self._getitem(index))
    finally:
        self._tiffpage = tiffpage
    self.pages[index] = self._keyframe