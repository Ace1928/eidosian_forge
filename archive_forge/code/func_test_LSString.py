import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
def test_LSString():
    lss = text.LSString('abc\ndef')
    assert lss.l == ['abc', 'def']
    assert lss.s == 'abc def'
    lss = text.LSString(os.getcwd())
    assert isinstance(lss.p[0], Path)