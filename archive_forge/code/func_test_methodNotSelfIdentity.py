from __future__ import annotations
import copyreg
import io
import pickle
import sys
import textwrap
from typing import Any, Callable, List, Tuple
from typing_extensions import NoReturn
from twisted.persisted import aot, crefutil, styles
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_methodNotSelfIdentity(self) -> None:
    """
        If a class change after an instance has been created,
        L{aot.unjellyFromSource} shoud raise a C{TypeError} when trying to
        unjelly the instance.
        """
    a = A()
    b = B()
    a.bmethod = b.bmethod
    b.a = a
    savedbmethod = B.bmethod
    del B.bmethod
    try:
        self.assertRaises(TypeError, aot.unjellyFromSource, aot.jellyToSource(b))
    finally:
        B.bmethod = savedbmethod