import platform
import time
import unittest
import pytest
from monty.functools import (
def test_invalidate_mangled_attribute(self):
    called = []

    class Bar:

        @cached
        def __bar(self):
            called.append('bar')
            return 1

        def get_bar(self):
            return self.__bar
    b = Bar()
    assert b.get_bar() == 1
    assert len(called) == 1
    cached.invalidate(b, '_Bar__bar')
    assert b.get_bar() == 1
    assert len(called) == 2