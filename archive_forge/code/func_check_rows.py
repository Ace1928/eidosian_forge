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
def check_rows(rows):
    """
            Yield each row in rows,
            but check each row first (for correct width).
            """
    for i, row in enumerate(rows):
        try:
            wrong_length = len(row) != vpr
        except TypeError:
            wrong_length = False
        if wrong_length:
            raise ProtocolError(f'Expected {vpr} values but got {len(row)} values, in row {i}')
        yield row