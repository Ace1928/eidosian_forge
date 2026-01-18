from __future__ import annotations
import functools
import operator
import re
from . import ExifTags, Image, ImagePalette
def cover(image, size, method=Image.Resampling.BICUBIC):
    """
    Returns a resized version of the image, so that the requested size is
    covered, while maintaining the original aspect ratio.

    :param image: The image to resize.
    :param size: The requested output size in pixels, given as a
                 (width, height) tuple.
    :param method: Resampling method to use. Default is
                   :py:attr:`~PIL.Image.Resampling.BICUBIC`.
                   See :ref:`concept-filters`.
    :return: An image.
    """
    im_ratio = image.width / image.height
    dest_ratio = size[0] / size[1]
    if im_ratio != dest_ratio:
        if im_ratio < dest_ratio:
            new_height = round(image.height / image.width * size[0])
            if new_height != size[1]:
                size = (size[0], new_height)
        else:
            new_width = round(image.width / image.height * size[1])
            if new_width != size[0]:
                size = (new_width, size[1])
    return image.resize(size, resample=method)