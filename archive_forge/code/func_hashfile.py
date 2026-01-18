import skvideo.io
import sys
import numpy as np
import hashlib
import os
from numpy.testing import assert_equal
def hashfile(afile, hasher, blocksize=65536):
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    return hasher.hexdigest()