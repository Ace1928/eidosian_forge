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
def itertrns(pixels):
    for row in pixels:
        row = group(row, planes)
        opa = map(it.__ne__, row)
        opa = map(maxval.__mul__, opa)
        opa = list(zip(opa))
        yield array(typecode, itertools.chain(*map(operator.add, row, opa)))