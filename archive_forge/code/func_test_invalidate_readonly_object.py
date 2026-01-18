import platform
import time
import unittest
import pytest
from monty.functools import (
def test_invalidate_readonly_object(self):
    called = []

    class Bar:
        __slots__ = ()

        @cached
        def bar(self):
            called.append('bar')
            return 1
    b = Bar()
    with pytest.raises(AttributeError, match="'Bar' object has no attribute '__dict__'"):
        cached.invalidate(b, 'bar')