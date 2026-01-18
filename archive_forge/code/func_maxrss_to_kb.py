import errno
import numbers
import os
import subprocess
import sys
from itertools import zip_longest
from io import UnsupportedOperation
def maxrss_to_kb(v):
    return int(v)