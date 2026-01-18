from __future__ import annotations
import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import Callable, MutableMapping
from enum import IntEnum
from pathlib import Path
from . import (
from ._binary import i32le, o32be, o32le
from ._util import DeferredError, is_path
def getexif(self):
    """
        Gets EXIF data from the image.

        :returns: an :py:class:`~PIL.Image.Exif` object.
        """
    if self._exif is None:
        self._exif = Exif()
        self._exif._loaded = False
    elif self._exif._loaded:
        return self._exif
    self._exif._loaded = True
    exif_info = self.info.get('exif')
    if exif_info is None:
        if 'Raw profile type exif' in self.info:
            exif_info = bytes.fromhex(''.join(self.info['Raw profile type exif'].split('\n')[3:]))
        elif hasattr(self, 'tag_v2'):
            self._exif.bigtiff = self.tag_v2._bigtiff
            self._exif.endian = self.tag_v2._endian
            self._exif.load_from_fp(self.fp, self.tag_v2._offset)
    if exif_info is not None:
        self._exif.load(exif_info)
    if ExifTags.Base.Orientation not in self._exif:
        xmp_tags = self.info.get('XML:com.adobe.xmp')
        if xmp_tags:
            match = re.search('tiff:Orientation(="|>)([0-9])', xmp_tags)
            if match:
                self._exif[ExifTags.Base.Orientation] = int(match[2])
    return self._exif