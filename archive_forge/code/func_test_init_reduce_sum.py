import numpy
import pytest
from thinc.api import reduce_first, reduce_last, reduce_max, reduce_mean, reduce_sum
from thinc.types import Ragged
def test_init_reduce_sum():
    model = reduce_sum()