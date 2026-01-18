import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_model_defaults_to_cpu(model_with_no_args):
    assert not isinstance(model_with_no_args.ops, CupyOps)