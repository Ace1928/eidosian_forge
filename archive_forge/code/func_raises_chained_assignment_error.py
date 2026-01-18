from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import (
import uuid
from pandas._config import using_copy_on_write
from pandas.compat import PYPY
from pandas.errors import ChainedAssignmentError
from pandas import set_option
from pandas.io.common import get_handle
def raises_chained_assignment_error(warn=True, extra_warnings=(), extra_match=()):
    from pandas._testing import assert_produces_warning
    if not warn:
        from contextlib import nullcontext
        return nullcontext()
    if PYPY and (not extra_warnings):
        from contextlib import nullcontext
        return nullcontext()
    elif PYPY and extra_warnings:
        return assert_produces_warning(extra_warnings, match='|'.join(extra_match))
    else:
        if using_copy_on_write():
            warning = ChainedAssignmentError
            match = 'A value is trying to be set on a copy of a DataFrame or Series through chained assignment'
        else:
            warning = FutureWarning
            match = 'ChainedAssignmentError'
        if extra_warnings:
            warning = (warning, *extra_warnings)
        return assert_produces_warning(warning, match='|'.join((match, *extra_match)))