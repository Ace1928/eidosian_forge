from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
def set_as_raw(self, data, rawmode=None):
    """
        Convenience method to set the internal image from a stream of raw data

        :param data: Bytes to be set
        :param rawmode: The rawmode to be used for the decoder.
            If not specified, it will default to the mode of the image
        :returns: None
        """
    if not rawmode:
        rawmode = self.mode
    d = Image._getdecoder(self.mode, 'raw', rawmode)
    d.setimage(self.im, self.state.extents())
    s = d.decode(data)
    if s[0] >= 0:
        msg = 'not enough image data'
        raise ValueError(msg)
    if s[1] != 0:
        msg = 'cannot decode image data'
        raise ValueError(msg)