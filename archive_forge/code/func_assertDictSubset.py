from __future__ import annotations
import sys
import warnings
from io import StringIO
from typing import Mapping, Sequence, TypeVar
from unittest import TestResult
from twisted.python.filepath import FilePath
from twisted.trial._synctest import (
from twisted.trial.unittest import SynchronousTestCase
import warnings
import warnings
def assertDictSubset(self, set: Mapping[_K, _V], subset: Mapping[_K, _V]) -> None:
    """
        Assert that all the keys present in C{subset} are also present in
        C{set} and that the corresponding values are equal.
        """
    for k, v in subset.items():
        self.assertEqual(set[k], v)