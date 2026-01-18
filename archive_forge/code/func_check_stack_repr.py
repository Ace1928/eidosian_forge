from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def check_stack_repr(gdb, expr, expected):
    """
    Check printing a stack-located value.
    """
    s = gdb.print_value(expr)
    if isinstance(expected, re.Pattern):
        assert expected.match(s), s
    else:
        assert s == expected