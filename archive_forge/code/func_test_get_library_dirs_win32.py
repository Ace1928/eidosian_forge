import os
import subprocess
import sys
import pytest
import pyarrow as pa
from pyarrow.lib import ArrowInvalid
@pytest.mark.skipif('sys.platform != "win32"')
def test_get_library_dirs_win32():
    assert any((os.path.exists(os.path.join(directory, 'arrow.lib')) for directory in pa.get_library_dirs()))