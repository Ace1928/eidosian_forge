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
def getchannel(self, channel):
    """
        Returns an image containing a single channel of the source image.

        :param channel: What channel to return. Could be index
          (0 for "R" channel of "RGB") or channel name
          ("A" for alpha channel of "RGBA").
        :returns: An image in "L" mode.

        .. versionadded:: 4.3.0
        """
    self.load()
    if isinstance(channel, str):
        try:
            channel = self.getbands().index(channel)
        except ValueError as e:
            msg = f'The image has no channel "{channel}"'
            raise ValueError(msg) from e
    return self._new(self.im.getband(channel))