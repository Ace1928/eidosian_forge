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
def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0