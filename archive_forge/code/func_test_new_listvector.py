import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
@pytest.mark.parametrize('vec', (robjects.ListVector({'a': 1, 'b': 2}), robjects.ListVector((('a', 1), ('b', 2))), robjects.ListVector(iter([('a', 1), ('b', 2)]))))
def test_new_listvector(vec):
    assert 'a' in vec.names
    assert 'b' in vec.names
    assert len(vec) == 2
    assert len(vec.names) == 2