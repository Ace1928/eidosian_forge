from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_method_descriptor():
    context = limited()
    assert guarded_eval('list.copy.__name__', context) == 'copy'