import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def convert_rgb_to_rgba(row, result):
    """
    Convert an RGB image to RGBA.
    This method assumes the alpha channel in result is
    already correctly initialized.
    """
    for i in range(3):
        result[i::4] = row[i::3]