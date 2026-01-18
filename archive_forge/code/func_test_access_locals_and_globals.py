from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_access_locals_and_globals():
    context = EvaluationContext(locals={'local_a': 'a'}, globals={'global_b': 'b'}, evaluation='limited')
    assert guarded_eval('local_a', context) == 'a'
    assert guarded_eval('global_b', context) == 'b'