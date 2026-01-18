import os
import subprocess
import sys
import pytest
import pyarrow as pa
from pyarrow.lib import ArrowInvalid
def test_get_include():
    include_dir = pa.get_include()
    assert os.path.exists(os.path.join(include_dir, 'arrow', 'api.h'))