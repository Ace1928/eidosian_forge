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
def test_hidden_callback(self):
    try:
        self.module.hidden_callback(2)
    except Exception as msg:
        assert str(msg).startswith('Callback global_f not defined')
    try:
        self.module.hidden_callback2(2)
    except Exception as msg:
        assert str(msg).startswith('cb: Callback global_f not defined')
    self.module.global_f = lambda x: x + 1
    r = self.module.hidden_callback(2)
    assert r == 3
    self.module.global_f = lambda x: x + 2
    r = self.module.hidden_callback(2)
    assert r == 4
    del self.module.global_f
    try:
        self.module.hidden_callback(2)
    except Exception as msg:
        assert str(msg).startswith('Callback global_f not defined')
    self.module.global_f = lambda x=0: x + 3
    r = self.module.hidden_callback(2)
    assert r == 5
    r = self.module.hidden_callback2(2)
    assert r == 3