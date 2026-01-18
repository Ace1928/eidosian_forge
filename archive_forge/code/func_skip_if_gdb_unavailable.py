from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def skip_if_gdb_unavailable():
    if not is_gdb_available():
        pytest.skip('gdb command unavailable')