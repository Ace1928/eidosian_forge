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
def test_access_type(self, tmp_path):
    fpath = util.getpath('tests', 'src', 'crackfortran', 'accesstype.f90')
    mod = crackfortran.crackfortran([str(fpath)])
    assert len(mod) == 1
    tt = mod[0]['vars']
    assert set(tt['a']['attrspec']) == {'private', 'bind(c)'}
    assert set(tt['b_']['attrspec']) == {'public', 'bind(c)'}
    assert set(tt['c']['attrspec']) == {'public'}