import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_grad_names():
    model = create_model('tmp')
    assert model.grad_names == tuple()
    model.set_param('param1', model.ops.alloc2f(4, 4))
    model.set_grad('param1', model.ops.alloc2f(4, 4) + 1)
    assert model.grad_names == ('param1',)