import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_model_can_save_to_disk(model_with_no_args):
    with make_tempdir() as path:
        model_with_no_args.to_disk(path / 'thinc_model')