from __future__ import annotations
from fontTools.misc.textTools import num2binary, binary2num, readHex, strjoin
import array
from io import StringIO
from typing import List
import re
import logging
def bitRepr(value, bits):
    s = ''
    for i in range(bits):
        s = '01'[value & 1] + s
        value = value >> 1
    return s