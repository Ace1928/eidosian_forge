from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_long_categorical_repr():
    cats = list('abcdefghijklmnopqrstuvwxyz')
    c = Categorical(cats, ordered=True)
    assert str(c) == 'categorical[[%s, ...], type=%s, ordered=True]' % (', '.join(map(repr, cats[:10])), c.type)