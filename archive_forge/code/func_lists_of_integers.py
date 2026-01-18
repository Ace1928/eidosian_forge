import numpy
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, integers, lists
from numpy.testing import assert_allclose
from thinc.layers import Embed
from thinc.layers.uniqued import uniqued
@composite
def lists_of_integers(draw, columns=2, lo=0, hi=ROWS - 1):
    int_list = draw(lists(integers(min_value=lo, max_value=hi)))
    int_list = int_list[len(int_list) % columns:]
    array = numpy.array(int_list, dtype='uint64')
    return array.reshape((-1, columns))