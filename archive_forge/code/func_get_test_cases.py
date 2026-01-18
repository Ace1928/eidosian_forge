from __future__ import annotations
import importlib.util
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING
import pytest
from numpy.typing.mypy_plugin import _EXTENDED_PRECISION_LIST
def get_test_cases(directory: str) -> Iterator[ParameterSet]:
    for root, _, files in os.walk(directory):
        for fname in files:
            short_fname, ext = os.path.splitext(fname)
            if ext in ('.pyi', '.py'):
                fullpath = os.path.join(root, fname)
                yield pytest.param(fullpath, id=short_fname)