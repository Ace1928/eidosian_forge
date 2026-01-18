import platform
import time
import unittest
import pytest
from monty.functools import (
def test_invalidate_nonlazy_private_attribute(self):
    called = []

    class Foo:

        def __foo(self):
            called.append('foo')
            return 1
    f = Foo()
    with pytest.raises(AttributeError, match="type object 'Foo' has no attribute 'foo'"):
        lazy_property.invalidate(f, 'foo')