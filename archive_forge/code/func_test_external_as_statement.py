import importlib
import codecs
import time
import unicodedata
import pytest
import numpy as np
from numpy.f2py.crackfortran import markinnerspaces, nameargspattern
from . import util
from numpy.f2py import crackfortran
import textwrap
import contextlib
import io
def test_external_as_statement(self):

    def incr(x):
        return x + 123
    r = self.module.external_as_statement(incr)
    assert r == 123