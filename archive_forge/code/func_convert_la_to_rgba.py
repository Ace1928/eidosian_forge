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
def convert_la_to_rgba(row, result):
    for i in range(3):
        result[i::4] = row[0::2]
    result[3::4] = row[1::2]