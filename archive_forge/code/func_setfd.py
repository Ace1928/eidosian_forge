from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
def setfd(self, fd):
    """
        Called from ImageFile to set the Python file-like object

        :param fd: A Python file-like object
        :returns: None
        """
    self.fd = fd