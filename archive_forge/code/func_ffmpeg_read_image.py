from __future__ import division
import logging
import os
import re
import subprocess as sp
import warnings
import numpy as np
from moviepy.compat import DEVNULL, PY3
from moviepy.config import get_setting  # ffmpeg, ffmpeg.exe, etc...
from moviepy.tools import cvsecs
def ffmpeg_read_image(filename, with_mask=True):
    """ Read an image file (PNG, BMP, JPEG...).

    Wraps FFMPEG_Videoreader to read just one image.
    Returns an ImageClip.

    This function is not meant to be used directly in MoviePy,
    use ImageClip instead to make clips out of image files.

    Parameters
    -----------

    filename
      Name of the image file. Can be of any format supported by ffmpeg.

    with_mask
      If the image has a transparency layer, ``with_mask=true`` will save
      this layer as the mask of the returned ImageClip

    """
    pix_fmt = 'rgba' if with_mask else 'rgb24'
    reader = FFMPEG_VideoReader(filename, pix_fmt=pix_fmt, check_duration=False)
    im = reader.lastread
    del reader
    return im