import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def smaller_types(self):
    bits = typeinfo[self.NAME].alignment
    types = []
    for name in _type_names:
        if typeinfo[name].alignment < bits:
            types.append(Type(name))
    return types