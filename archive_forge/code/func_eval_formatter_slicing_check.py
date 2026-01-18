import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
def eval_formatter_slicing_check(f):
    ns = dict(n=12, pi=math.pi, stuff='hello there', os=os)
    s = f.format(' {stuff.split()[:]} ', **ns)
    assert s == " ['hello', 'there'] "
    s = f.format(' {stuff.split()[::-1]} ', **ns)
    assert s == " ['there', 'hello'] "
    s = f.format('{stuff[::2]}', **ns)
    assert s == ns['stuff'][::2]
    pytest.raises(SyntaxError, f.format, '{n:x}', **ns)