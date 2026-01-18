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
@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason='Mypy is not installed')
@pytest.mark.parametrize('path', get_test_cases(PASS_DIR))
def test_code_runs(path: str) -> None:
    """Validate that the code in `path` properly during runtime."""
    path_without_extension, _ = os.path.splitext(path)
    dirname, filename = path.split(os.sep)[-2:]
    spec = importlib.util.spec_from_file_location(f'{dirname}.{filename}', path)
    assert spec is not None
    assert spec.loader is not None
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)