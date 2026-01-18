import os
import sys
from mmap import mmap, ACCESS_READ
from mmap import ALLOCATIONGRANULARITY
def is_64_bit():
    """:return: True if the system is 64 bit. Otherwise it can be assumed to be 32 bit"""
    return sys.maxsize > (1 << 32) - 1