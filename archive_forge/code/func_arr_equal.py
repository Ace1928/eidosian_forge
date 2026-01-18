import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def arr_equal(self, arr1, arr2):
    if arr1.shape != arr2.shape:
        return False
    return (arr1 == arr2).all()