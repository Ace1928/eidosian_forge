from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def hexToInt(hex_str):
    return int(hex_str, 16)