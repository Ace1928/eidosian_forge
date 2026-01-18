from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def select_frame(self, func_name):
    """
        Select the innermost frame with the given function name.
        """
    out = self.run_command('info stack')
    pat = '(?mi)^#(\\d+)\\s+.* in ' + re.escape(func_name) + '\\b'
    m = re.search(pat, out)
    if m is None:
        pytest.fail(f'Could not select frame for function {func_name}')
    frame_num = int(m[1])
    out = self.run_command(f'frame {frame_num}')
    assert f'in {func_name}' in out