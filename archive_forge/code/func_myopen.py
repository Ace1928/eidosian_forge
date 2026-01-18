from __future__ import annotations
import builtins
import io
import os
import sys
import pytest
from dask.system import cpu_count
def myopen(path, *args, **kwargs):
    if path in paths:
        return paths.get(path)
    return builtin_open(path, *args, **kwargs)