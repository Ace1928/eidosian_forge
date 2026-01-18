import io
import numpy as np
import pytest
from pandas import (
def bar_grad(a=None, b=None, c=None, d=None):
    """Used in multiple tests to simplify formatting of expected result"""
    ret = [('width', '10em')]
    if all((x is None for x in [a, b, c, d])):
        return ret
    return ret + [('background', f'linear-gradient(90deg,{','.join([x for x in [a, b, c, d] if x])})')]