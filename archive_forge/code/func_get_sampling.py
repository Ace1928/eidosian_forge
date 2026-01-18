from __future__ import annotations
import array
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from .JpegPresets import presets
def get_sampling(im):
    if not hasattr(im, 'layers') or im.layers in (1, 4):
        return -1
    sampling = im.layer[0][1:3] + im.layer[1][1:3] + im.layer[2][1:3]
    return samplings.get(sampling, -1)