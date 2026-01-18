from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
def register_init(name, model, X=None, Y=None):
    init_was_called[name] = True