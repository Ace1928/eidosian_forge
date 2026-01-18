import math
import textwrap
import sys
import pytest
import threading
import traceback
import time
import numpy as np
from numpy.testing import IS_PYPY
from . import util
def test_gh18335(self):

    def bar(x):
        return x * x
    res = self.module.foo(bar)
    assert res == 110