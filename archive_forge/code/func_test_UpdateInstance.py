from __future__ import annotations
import os
import sys
import types
from typing_extensions import NoReturn
from twisted.python import rebuild
from twisted.trial.unittest import TestCase
from . import crash_test_dummy
def test_UpdateInstance(self) -> None:
    global Foo, Buz
    b = Buz()

    class Foo:

        def foo(self) -> None:
            """
                Dummy method
                """

    class Buz(Bar, Baz):
        x = 10
    rebuild.updateInstance(b)
    assert hasattr(b, 'foo'), 'Missing method on rebuilt instance'
    assert hasattr(b, 'x'), 'Missing class attribute on rebuilt instance'