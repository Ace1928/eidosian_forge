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
def iterpal(pixels):
    for row in pixels:
        row = [plte[x] for x in row]
        yield array('B', itertools.chain(*row))