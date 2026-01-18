import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
def test_dollar_formatter():
    f = text.DollarFormatter()
    eval_formatter_check(f)
    eval_formatter_slicing_check(f)
    ns = dict(n=12, pi=math.pi, stuff='hello there', os=os)
    s = f.format('$n', **ns)
    assert s == '12'
    s = f.format('$n.real', **ns)
    assert s == '12'
    s = f.format('$n/{stuff[:5]}', **ns)
    assert s == '12/hello'
    s = f.format('$n $$HOME', **ns)
    assert s == '12 $HOME'
    s = f.format('${foo}', foo='HOME')
    assert s == '$HOME'