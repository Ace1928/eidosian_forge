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
def test_methodSelfIdentity(self) -> None:
    a = A()
    b = B()
    a.bmethod = b.bmethod
    b.a = a
    im_ = aot.unjellyFromSource(aot.jellyToSource(b)).a.bmethod
    self.assertEqual(aot._selfOfMethod(im_).__class__, aot._classOfMethod(im_))