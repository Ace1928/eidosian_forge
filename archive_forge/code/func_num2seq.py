import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
@property
def num2seq(self):
    if self.type.NAME.startswith('STRING'):
        elsize = self.type.elsize
        return ['1' * elsize, '2' * elsize]
    return [1, 2]