import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def flags2names(flags):
    info = []
    for flagname in ['CONTIGUOUS', 'FORTRAN', 'OWNDATA', 'ENSURECOPY', 'ENSUREARRAY', 'ALIGNED', 'NOTSWAPPED', 'WRITEABLE', 'WRITEBACKIFCOPY', 'UPDATEIFCOPY', 'BEHAVED', 'BEHAVED_RO', 'CARRAY', 'FARRAY']:
        if abs(flags) & getattr(wrap, flagname, 0):
            info.append(flagname)
    return info