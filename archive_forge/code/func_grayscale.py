from __future__ import annotations
import functools
import operator
import re
from . import ExifTags, Image, ImagePalette
def grayscale(image):
    """
    Convert the image to grayscale.

    :param image: The image to convert.
    :return: An image.
    """
    return image.convert('L')