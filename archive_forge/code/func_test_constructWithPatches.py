from __future__ import annotations
from typing_extensions import NoReturn
from twisted.python.monkey import MonkeyPatcher
from twisted.trial import unittest
def test_constructWithPatches(self) -> None:
    """
        Constructing a L{MonkeyPatcher} with patches should add all of the
        given patches to the patch list.
        """
    patcher = MonkeyPatcher((self.testObject, 'foo', 'haha'), (self.testObject, 'bar', 'hehe'))
    patcher.patch()
    self.assertEqual('haha', self.testObject.foo)
    self.assertEqual('hehe', self.testObject.bar)
    self.assertEqual(self.originalObject.baz, self.testObject.baz)