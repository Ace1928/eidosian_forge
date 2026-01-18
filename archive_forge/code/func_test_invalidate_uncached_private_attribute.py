import platform
import time
import unittest
import pytest
from monty.functools import (
def test_invalidate_uncached_private_attribute(self):
    called = []

    class Bar:

        def __bar(self):
            called.append('bar')
            return 1
    b = Bar()
    with pytest.raises(AttributeError, match="'Bar._Bar__bar' is not a cached attribute"):
        cached.invalidate(b, '__bar')