from __future__ import annotations
import builtins
import io
import os
import sys
import pytest
from dask.system import cpu_count
def mycpu_count():
    return 250