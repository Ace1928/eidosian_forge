import os
import subprocess
import sys
import pytest
import pyarrow as pa
from pyarrow.lib import ArrowInvalid
def test_import_at_shutdown():
    code = 'if 1:\n        import atexit\n\n        def import_arrow():\n            import pyarrow\n\n        atexit.register(import_arrow)\n        '
    subprocess.check_call([sys.executable, '-c', code])