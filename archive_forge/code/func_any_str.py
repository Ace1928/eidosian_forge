import sys
from contextlib import contextmanager
from typing import (
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def any_str(self, x: AnyStr) -> AnyStr:
    return x