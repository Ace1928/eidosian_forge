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
def test_extended_precision() -> None:
    path = os.path.join(MISC_DIR, 'extended_precision.pyi')
    output_mypy = OUTPUT_MYPY
    assert path in output_mypy
    with open(path) as f:
        expression_list = f.readlines()
    for _msg in output_mypy[path]:
        lineno, msg = _strip_filename(_msg)
        expression = expression_list[lineno - 1].rstrip('\n')
        if LINENO_MAPPING[lineno] in _EXTENDED_PRECISION_LIST:
            raise AssertionError(_REVEAL_MSG.format(lineno, msg))
        elif 'error' not in msg:
            _test_fail(path, expression, msg, 'Expression is of type "Any"', lineno)