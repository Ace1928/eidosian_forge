from __future__ import annotations
from typing import Callable, Iterable
from typing_extensions import Concatenate, ParamSpec
from twisted.python import formmethod
from twisted.trial import unittest
def testFlags(self) -> None:
    flags = [('a', 'apple', 'an apple'), ('b', 'banana', 'ook')]
    self.argTest(formmethod.Flags, [(['a'], ['apple']), (['b', 'a'], ['banana', 'apple'])], (['a', 'c'], ['fdfs']), flags=flags)