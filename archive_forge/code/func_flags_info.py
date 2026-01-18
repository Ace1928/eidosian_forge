import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def flags_info(arr):
    flags = wrap.array_attrs(arr)[6]
    return flags2names(flags)