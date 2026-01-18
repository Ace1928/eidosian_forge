import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_model_can_load_from_disk(model_with_no_args):
    with make_tempdir() as path:
        model_with_no_args.to_disk(path / 'thinc_model')
        m2 = model_with_no_args.from_disk(path / 'thinc_model')
    assert model_with_no_args.to_bytes() == m2.to_bytes()