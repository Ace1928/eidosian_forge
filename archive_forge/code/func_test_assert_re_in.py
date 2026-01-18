import os
import sys
import warnings
import numpy as np
import pytest
from ..casting import sctypes
from ..testing import (
@pytest.mark.parametrize('regex, entries', [['.*', ''], ['.*', ['any']], ['ab', 'abc'], ['ab', ['', 'abc', 'laskdjf']], ['ab', ('', 'abc', 'laskdjf')], pytest.param('ab', 'cab', marks=pytest.mark.xfail), pytest.param('ab$', 'abc', marks=pytest.mark.xfail), pytest.param('ab$', ['ddd', ''], marks=pytest.mark.xfail), pytest.param('ab$', ('ddd', ''), marks=pytest.mark.xfail), pytest.param('', [], marks=pytest.mark.xfail)])
def test_assert_re_in(regex, entries):
    assert_re_in(regex, entries)