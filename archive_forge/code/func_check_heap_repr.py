from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def check_heap_repr(gdb, expr, expected):
    """
    Check printing a heap-located value, given its address.
    """
    s = gdb.print_value(f'*{expr}')
    if s != expected:
        assert s.endswith(f' {expected}')