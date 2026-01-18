import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_list_repr():
    vec = robjects.vectors.ListVector((('a', 1), ('b', 2), ('b', 3)))
    s = repr(vec)
    assert s.startswith('<rpy2.robjects.vectors.ListVector ')
    vec2 = robjects.vectors.ListVector((('A', vec),))
    s = repr(vec2)
    assert s.startswith('<rpy2.robjects.vectors.ListVector ')