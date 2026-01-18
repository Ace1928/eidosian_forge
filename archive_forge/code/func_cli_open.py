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
def cli_open(path):
    if path == '-':
        return binary_stdin()
    return open(path, 'rb')