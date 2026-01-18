import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
def eval_formatter_check(f):
    ns = dict(n=12, pi=math.pi, stuff='hello there', os=os, u=u'café', b='café')
    s = f.format('{n} {n//4} {stuff.split()[0]}', **ns)
    assert s == '12 3 hello'
    s = f.format(' '.join(['{n//%i}' % i for i in range(1, 8)]), **ns)
    assert s == '12 6 4 3 2 2 1'
    s = f.format('{[n//i for i in range(1,8)]}', **ns)
    assert s == '[12, 6, 4, 3, 2, 2, 1]'
    s = f.format('{stuff!s}', **ns)
    assert s == ns['stuff']
    s = f.format('{stuff!r}', **ns)
    assert s == repr(ns['stuff'])
    s = f.format('{u}', **ns)
    assert s == ns['u']
    s = f.format('{b}', **ns)
    pytest.raises(NameError, f.format, '{dne}', **ns)