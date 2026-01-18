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
def iterrgb():
    for row in pixels:
        a = newarray() * 3 * width
        for i in range(3):
            a[i::3] = row
        yield a