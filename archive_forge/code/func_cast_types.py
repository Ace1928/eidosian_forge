import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def cast_types(self):
    return [self.__class__(_m) for _m in _cast_dict[self.NAME]]