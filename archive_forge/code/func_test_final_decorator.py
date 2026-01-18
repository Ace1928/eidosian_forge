import signal
import sys
import types
from typing import Any, TypeVar
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
from .. import _core
from .._core._tests.tutil import (
from .._util import (
from ..testing import wait_all_tasks_blocked
def test_final_decorator() -> None:
    """Test that subclassing a @final-annotated class is not allowed.

    This checks both runtime results, and verifies that type checkers detect
    the error statically through the type-ignore comment.
    """

    @final
    class FinalClass:
        pass
    with pytest.raises(TypeError):

        class SubClass(FinalClass):
            pass