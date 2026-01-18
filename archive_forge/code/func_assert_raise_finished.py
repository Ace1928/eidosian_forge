import array
import base64
import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import cairocffi
import pikepdf
import pytest
from . import (
def assert_raise_finished(func, *args, **kwargs):
    with pytest.raises(cairocffi.CairoError) as exc:
        func(*args, **kwargs)
    assert 'SURFACE_FINISHED' in str(exc) or 'ExceptionInfo' in str(exc)