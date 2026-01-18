from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def jpeg_decode_colorspace(photometric: int, planarconfig: int, extrasamples: tuple[int, ...], jfif: bool, /) -> tuple[int | None, int | str | None]:
    """Return JPEG and output color space for `jpeg_decode` function."""
    colorspace: int | None = None
    outcolorspace: int | str | None = None
    if extrasamples:
        pass
    elif photometric == 6:
        outcolorspace = 2
    elif photometric == 2:
        if not jfif:
            colorspace = 2
        outcolorspace = 2
    elif photometric == 5:
        outcolorspace = 4
    elif photometric > 3:
        outcolorspace = PHOTOMETRIC(photometric).name
    if planarconfig != 1:
        outcolorspace = 1
    return (colorspace, outcolorspace)