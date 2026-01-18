from io import BytesIO
import os
import pathlib
import tarfile
import zipfile
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
@pytest.fixture
def gcs_buffer():
    """Emulate GCS using a binary buffer."""
    pytest.importorskip('gcsfs')
    fsspec = pytest.importorskip('fsspec')
    gcs_buffer = BytesIO()
    gcs_buffer.close = lambda: True

    class MockGCSFileSystem(fsspec.AbstractFileSystem):

        @staticmethod
        def open(*args, **kwargs):
            gcs_buffer.seek(0)
            return gcs_buffer

        def ls(self, path, **kwargs):
            return [{'name': path, 'type': 'file'}]
    fsspec.register_implementation('gs', MockGCSFileSystem, clobber=True)
    return gcs_buffer