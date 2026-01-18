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
@pytest.mark.parametrize('dimspec', all_dimspecs)
def test_inv_array_size(self, dimspec):
    count = self.all_dimspecs.index(dimspec)
    get_arr_size = getattr(self.module, f'get_arr_size_{count}')
    get_inv_arr_size = getattr(self.module, f'get_inv_arr_size_{count}')
    for n in [1, 2, 3, 4, 5]:
        sz, a = get_arr_size(n)
        if dimspec in self.nonlinear_dimspecs:
            n1 = get_inv_arr_size(a, n)
        else:
            n1 = get_inv_arr_size(a)
        sz1, _ = get_arr_size(n1)
        assert sz == sz1, (n, n1, sz, sz1)