import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_maybe_methods():
    model = Linear(5)
    assert model.maybe_get_dim('nI') is None
    model.set_dim('nI', 4)
    assert model.maybe_get_dim('nI') == 4
    assert model.maybe_get_ref('boo') is None
    assert model.maybe_get_param('W') is None
    model.initialize()
    assert model.maybe_get_param('W') is not None