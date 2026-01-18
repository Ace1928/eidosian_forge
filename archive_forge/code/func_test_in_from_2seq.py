import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_in_from_2seq(self):
    a = self.array([2], intent.in_, self.num2seq)
    assert not a.has_shared_memory()