import platform
import time
import unittest
import pytest
from monty.functools import (
def test_invalidate_superclass_attribute(self):
    called = []

    class Bar:

        @lazy_property
        def bar(self):
            called.append('bar')
            return 1
    b = Bar()
    with pytest.raises(AttributeError, match="'Bar.bar' is not a cached attribute"):
        cached.invalidate(b, 'bar')