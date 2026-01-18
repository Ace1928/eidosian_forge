from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def create_context(evaluation: str, **kwargs):
    return EvaluationContext(locals=kwargs, globals={}, evaluation=evaluation)