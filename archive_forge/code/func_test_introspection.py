import platform
import time
import unittest
import pytest
from monty.functools import (
def test_introspection(self):

    class Foo:

        def foo(self):
            """foo func doc"""

        @lazy_property
        def bar(self):
            """bar func doc"""
    assert Foo.foo.__name__ == 'foo'
    assert Foo.foo.__doc__ == 'foo func doc'
    assert 'test_functools' in Foo.foo.__module__
    assert Foo.bar.__name__ == 'bar'
    assert Foo.bar.__doc__ == 'bar func doc'
    assert 'test_functools' in Foo.bar.__module__