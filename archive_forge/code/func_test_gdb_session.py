from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_gdb_session(gdb):
    out = gdb.run_command('show version')
    assert out.startswith('GNU gdb ('), out