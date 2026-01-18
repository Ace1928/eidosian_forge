from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@contextmanager
def module_not_installed(module: str):
    import sys
    try:
        to_restore = sys.modules[module]
        del sys.modules[module]
    except KeyError:
        to_restore = None
    try:
        yield
    finally:
        sys.modules[module] = to_restore