import platform
import time
import unittest
import pytest
from monty.functools import (
def test_evaluate_once(self):
    called = []

    class Foo:

        @lazy_property
        def foo(self):
            called.append('foo')
            return 1
    f = Foo()
    assert f.foo == 1
    assert f.foo == 1
    assert f.foo == 1
    assert len(called) == 1