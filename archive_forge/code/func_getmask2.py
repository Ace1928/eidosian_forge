from __future__ import annotations
import base64
import os
import sys
import warnings
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from . import Image
from ._util import is_directory, is_path
def getmask2(self, text, mode='', direction=None, features=None, language=None, stroke_width=0, anchor=None, ink=0, start=None, *args, **kwargs):
    """
        Create a bitmap for the text.

        If the font uses antialiasing, the bitmap should have mode ``L`` and use a
        maximum value of 255. If the font has embedded color data, the bitmap
        should have mode ``RGBA``. Otherwise, it should have mode ``1``.

        :param text: Text to render.
        :param mode: Used by some graphics drivers to indicate what mode the
                     driver prefers; if empty, the renderer may return either
                     mode. Note that the mode is always a string, to simplify
                     C-level implementations.

                     .. versionadded:: 1.1.5

        :param direction: Direction of the text. It can be 'rtl' (right to
                          left), 'ltr' (left to right) or 'ttb' (top to bottom).
                          Requires libraqm.

                          .. versionadded:: 4.2.0

        :param features: A list of OpenType font features to be used during text
                         layout. This is usually used to turn on optional
                         font features that are not enabled by default,
                         for example 'dlig' or 'ss01', but can be also
                         used to turn off default font features for
                         example '-liga' to disable ligatures or '-kern'
                         to disable kerning.  To get all supported
                         features, see
                         https://learn.microsoft.com/en-us/typography/opentype/spec/featurelist
                         Requires libraqm.

                         .. versionadded:: 4.2.0

        :param language: Language of the text. Different languages may use
                         different glyph shapes or ligatures. This parameter tells
                         the font which language the text is in, and to apply the
                         correct substitutions as appropriate, if available.
                         It should be a `BCP 47 language code
                         <https://www.w3.org/International/articles/language-tags/>`_
                         Requires libraqm.

                         .. versionadded:: 6.0.0

        :param stroke_width: The width of the text stroke.

                         .. versionadded:: 6.2.0

        :param anchor:  The text anchor alignment. Determines the relative location of
                        the anchor to the text. The default alignment is top left,
                        specifically ``la`` for horizontal text and ``lt`` for
                        vertical text. See :ref:`text-anchors` for details.

                         .. versionadded:: 8.0.0

        :param ink: Foreground ink for rendering in RGBA mode.

                         .. versionadded:: 8.0.0

        :param start: Tuple of horizontal and vertical offset, as text may render
                      differently when starting at fractional coordinates.

                         .. versionadded:: 9.4.0

        :return: A tuple of an internal PIL storage memory instance as defined by the
                 :py:mod:`PIL.Image.core` interface module, and the text offset, the
                 gap between the starting coordinate and the first marking
        """
    _string_length_check(text)
    if start is None:
        start = (0, 0)
    im = None
    size = None

    def fill(width, height):
        nonlocal im, size
        size = (width, height)
        if Image.MAX_IMAGE_PIXELS is not None:
            pixels = max(1, width) * max(1, height)
            if pixels > 2 * Image.MAX_IMAGE_PIXELS:
                return
        im = Image.core.fill('RGBA' if mode == 'RGBA' else 'L', size)
        return im
    offset = self.font.render(text, fill, mode, direction, features, language, stroke_width, anchor, ink, start[0], start[1])
    Image._decompression_bomb_check(size)
    return (im, offset)