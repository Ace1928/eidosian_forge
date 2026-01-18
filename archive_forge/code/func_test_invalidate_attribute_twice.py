import platform
import time
import unittest
import pytest
from monty.functools import (
def test_invalidate_attribute_twice(self):
    called = []

    class Bar:

        @cached
        def bar(self):
            called.append('bar')
            return 1
    b = Bar()
    assert b.bar == 1
    assert len(called) == 1
    cached.invalidate(b, 'bar')
    cached.invalidate(b, 'bar')
    assert b.bar == 1
    assert len(called) == 2