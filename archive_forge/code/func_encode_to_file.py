from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
def encode_to_file(self, fh, bufsize):
    """
        :param fh: File handle.
        :param bufsize: Buffer size.

        :returns: If finished successfully, return 0.
            Otherwise, return an error code. Err codes are from
            :data:`.ImageFile.ERRORS`.
        """
    errcode = 0
    while errcode == 0:
        status, errcode, buf = self.encode(bufsize)
        if status > 0:
            fh.write(buf[status:])
    return errcode