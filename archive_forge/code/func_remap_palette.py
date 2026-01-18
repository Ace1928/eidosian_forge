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
def remap_palette(self, dest_map, source_palette=None):
    """
        Rewrites the image to reorder the palette.

        :param dest_map: A list of indexes into the original palette.
           e.g. ``[1,0]`` would swap a two item palette, and ``list(range(256))``
           is the identity transform.
        :param source_palette: Bytes or None.
        :returns:  An :py:class:`~PIL.Image.Image` object.

        """
    from . import ImagePalette
    if self.mode not in ('L', 'P'):
        msg = 'illegal image mode'
        raise ValueError(msg)
    bands = 3
    palette_mode = 'RGB'
    if source_palette is None:
        if self.mode == 'P':
            self.load()
            palette_mode = self.im.getpalettemode()
            if palette_mode == 'RGBA':
                bands = 4
            source_palette = self.im.getpalette(palette_mode, palette_mode)
        else:
            source_palette = bytearray((i // 3 for i in range(768)))
    palette_bytes = b''
    new_positions = [0] * 256
    for i, oldPosition in enumerate(dest_map):
        palette_bytes += source_palette[oldPosition * bands:oldPosition * bands + bands]
        new_positions[oldPosition] = i
    mapping_palette = bytearray(new_positions)
    m_im = self.copy()
    m_im._mode = 'P'
    m_im.palette = ImagePalette.ImagePalette(palette_mode, palette=mapping_palette * bands)
    m_im.im.putpalette(palette_mode + ';L', m_im.palette.tobytes())
    m_im = m_im.convert('L')
    m_im.putpalette(palette_bytes, palette_mode)
    m_im.palette = ImagePalette.ImagePalette(palette_mode, palette=palette_bytes)
    if 'transparency' in self.info:
        try:
            m_im.info['transparency'] = dest_map.index(self.info['transparency'])
        except ValueError:
            if 'transparency' in m_im.info:
                del m_im.info['transparency']
    return m_im