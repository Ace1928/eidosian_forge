import collections
from functools import partial
import string
import subprocess
import sys
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.core import ops
import pandas.core.common as com
from pandas.util.version import Version
def test_get_callable_name():
    getname = com.get_callable_name

    def fn(x):
        return x
    lambda_ = lambda x: x
    part1 = partial(fn)
    part2 = partial(part1)

    class somecall:

        def __call__(self):
            raise NotImplementedError
    assert getname(fn) == 'fn'
    assert getname(lambda_)
    assert getname(part1) == 'fn'
    assert getname(part2) == 'fn'
    assert getname(somecall()) == 'somecall'
    assert getname(1) is None