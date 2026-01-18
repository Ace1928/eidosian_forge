import pytest
import contextlib
import os
import tempfile
from rpy2.robjects.packages import importr, data
from rpy2.robjects import r
from rpy2.robjects.lib import grdevices
@contextlib.contextmanager
def set_filenames_to_delete():
    todelete = set()
    yield todelete
    for fn in todelete:
        if os.path.exists(fn):
            os.unlink(fn)