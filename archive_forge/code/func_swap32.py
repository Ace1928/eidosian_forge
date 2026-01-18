from __future__ import annotations
import ctypes
import os
import sys
def swap32(x):
    return x << 24 & 4278190080 | x << 8 & 16711680 | x >> 8 & 65280 | x >> 24 & 255