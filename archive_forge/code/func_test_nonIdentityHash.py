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
def test_nonIdentityHash(self) -> None:
    global ClassWithCustomHash

    class ClassWithCustomHash(styles.Versioned):

        def __init__(self, unique: str, hash: int) -> None:
            self.unique = unique
            self.hash = hash

        def __hash__(self) -> int:
            return self.hash
    v1 = ClassWithCustomHash('v1', 0)
    v2 = ClassWithCustomHash('v2', 0)
    pkl = pickle.dumps((v1, v2))
    del v1, v2
    ClassWithCustomHash.persistenceVersion = 1
    ClassWithCustomHash.upgradeToVersion1 = lambda self: setattr(self, 'upgraded', True)
    v1, v2 = pickle.loads(pkl)
    styles.doUpgrade()
    self.assertEqual(v1.unique, 'v1')
    self.assertEqual(v2.unique, 'v2')
    self.assertTrue(v1.upgraded)
    self.assertTrue(v2.upgraded)